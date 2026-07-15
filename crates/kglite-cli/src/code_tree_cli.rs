//! CLI wrapper around the pure-Rust code-tree builder.

use std::ffi::OsString;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;

use anyhow::{Context, Result};
use clap::{Args, Subcommand, ValueEnum};
use kglite::api::code_tree::{archive_and_build, build_code_tree, build_code_tree_revs};
use kglite::api::io::save_graph;
use kglite::api::DirGraph;
use serde_json::{json, Value};

const METADATA_FORMAT: u64 = 1;
const DEFAULT_GRAPH: &str = ".kglite/code-review.kgl";

#[derive(Subcommand, Debug)]
pub enum CodeTreeCommand {
    /// Parse a checkout or one or more git revisions into a `.kgl` graph.
    Build(BuildArgs),
    /// Check whether a built graph still matches its recorded source state.
    Status(StatusArgs),
}

#[derive(Args, Debug)]
pub struct BuildArgs {
    /// Source directory or project manifest to parse.
    pub source: PathBuf,
    /// Artifact path. Defaults to `<source>/.kglite/code-review.kgl`.
    #[arg(short, long)]
    pub output: Option<PathBuf>,
    /// Build committed content at one git revision.
    #[arg(long, conflicts_with = "revs")]
    pub rev: Option<String>,
    /// Merge committed content from several revisions, oldest to newest.
    #[arg(long, num_args = 1.., conflicts_with = "rev")]
    pub revs: Vec<String>,
    /// Override the git repository root used for revision builds.
    #[arg(long)]
    pub repo_root: Option<PathBuf>,
    /// Omit manifest-declared test roots.
    #[arg(long)]
    pub no_tests: bool,
    /// Include markdown documentation nodes and links.
    #[arg(long)]
    pub include_docs: bool,
    /// Skip parsing files above this line count while keeping File nodes.
    #[arg(long)]
    pub max_loc_per_file: Option<usize>,
    /// Print parser progress to stderr.
    #[arg(long)]
    pub verbose: bool,
    /// Status output format.
    #[arg(long, value_enum, default_value_t = StatusFormat::Human)]
    pub format: StatusFormat,
}

#[derive(Args, Debug)]
pub struct StatusArgs {
    /// Graph artifact whose sidecar should be checked.
    #[arg(short, long, default_value = DEFAULT_GRAPH)]
    pub output: PathBuf,
    /// Status output format.
    #[arg(long, value_enum, default_value_t = StatusFormat::Human)]
    pub format: StatusFormat,
}

#[derive(Clone, Copy, Debug, Default, ValueEnum)]
pub enum StatusFormat {
    #[default]
    Human,
    Json,
}

pub fn run(command: &CodeTreeCommand) -> Result<()> {
    match command {
        CodeTreeCommand::Build(args) => {
            let status = build(args)?;
            print_status(&status, args.format);
        }
        CodeTreeCommand::Status(args) => {
            let status = status(&args.output)?;
            print_status(&status, args.format);
        }
    }
    Ok(())
}

struct BuildPlan {
    source: PathBuf,
    output: PathBuf,
    repo_root: Option<PathBuf>,
    include_tests: bool,
    mode: &'static str,
    revisions: Vec<String>,
}

fn build(args: &BuildArgs) -> Result<Value> {
    let plan = prepare_build(args)?;
    let graph = construct_graph(args, &plan)?;
    persist_build(args, &plan, graph)
}

fn prepare_build(args: &BuildArgs) -> Result<BuildPlan> {
    let source = args
        .source
        .canonicalize()
        .with_context(|| format!("source does not exist: {}", args.source.display()))?;
    let output = resolved_output(&source, args.output.as_deref());
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("could not create {}", parent.display()))?;
    }
    let repo_root = args
        .repo_root
        .as_ref()
        .map(|p| p.canonicalize().unwrap_or_else(|_| p.clone()));
    let include_tests = !args.no_tests;
    let (mode, revisions) = if let Some(rev) = &args.rev {
        ("single-revision", vec![rev.clone()])
    } else if !args.revs.is_empty() {
        ("multi-revision", args.revs.clone())
    } else {
        ("working-tree", Vec::new())
    };

    Ok(BuildPlan {
        source,
        output,
        repo_root,
        include_tests,
        mode,
        revisions,
    })
}

fn construct_graph(args: &BuildArgs, plan: &BuildPlan) -> Result<Arc<DirGraph>> {
    let mut graph = match (args.rev.as_deref(), args.revs.is_empty()) {
        (Some(rev), _) => archive_and_build(
            &plan.source,
            rev,
            plan.repo_root.as_deref(),
            args.verbose,
            plan.include_tests,
            None,
            args.max_loc_per_file,
            args.include_docs,
        ),
        (None, false) => build_code_tree_revs(
            &plan.source,
            &args.revs,
            plan.repo_root.as_deref(),
            args.verbose,
            plan.include_tests,
            None,
            args.max_loc_per_file,
            args.include_docs,
        ),
        (None, true) => build_code_tree(
            &plan.source,
            args.verbose,
            plan.include_tests,
            None,
            args.max_loc_per_file,
            args.include_docs,
        ),
    }
    .map_err(anyhow::Error::msg)?;

    if plan.mode == "working-tree" {
        let instructions = format!(
            "Code graph built from the current working tree at {}. Refresh the artifact after source changes and verify review findings against exact source lines.",
            plan.source.display()
        );
        Arc::make_mut(&mut graph).set_instructions(&instructions, None);
    }
    Ok(graph)
}

fn persist_build(args: &BuildArgs, plan: &BuildPlan, mut graph: Arc<DirGraph>) -> Result<Value> {
    let output_text = plan.output.to_string_lossy().to_string();
    save_graph(&mut graph, &output_text)
        .map_err(|e| anyhow::anyhow!("failed to save {}: {e}", plan.output.display()))?;

    let fingerprint = source_fingerprint(&plan.source, plan.repo_root.as_deref(), &plan.revisions)?;
    let metadata = json!({
        "format": METADATA_FORMAT,
        "source": &plan.source,
        "output": &plan.output,
        "mode": plan.mode,
        "revisions": &plan.revisions,
        "repo_root": &plan.repo_root,
        "include_tests": plan.include_tests,
        "include_docs": args.include_docs,
        "max_loc_per_file": args.max_loc_per_file,
        "fingerprint": fingerprint,
    });
    let metadata_path = metadata_path(&plan.output);
    fs::write(&metadata_path, serde_json::to_vec_pretty(&metadata)?)
        .with_context(|| format!("could not write {}", metadata_path.display()))?;
    let bytes = fs::metadata(&plan.output)?.len();
    Ok(json!({
        "status": "built",
        "fresh": true,
        "graph": &plan.output,
        "metadata": metadata_path,
        "source": &plan.source,
        "mode": plan.mode,
        "revisions": metadata["revisions"],
        "bytes": bytes,
    }))
}

fn status(output: &Path) -> Result<Value> {
    let output = output
        .canonicalize()
        .unwrap_or_else(|_| output.to_path_buf());
    let sidecar = metadata_path(&output);
    if !output.exists() || !sidecar.exists() {
        return Ok(json!({
            "status": "missing",
            "fresh": false,
            "graph": output,
            "metadata": sidecar,
            "reason": "graph artifact or metadata sidecar is missing",
        }));
    }
    let metadata: Value = serde_json::from_slice(
        &fs::read(&sidecar).with_context(|| format!("could not read {}", sidecar.display()))?,
    )
    .with_context(|| format!("invalid metadata sidecar: {}", sidecar.display()))?;
    if metadata["format"].as_u64() != Some(METADATA_FORMAT) {
        return Ok(json!({
            "status": "stale",
            "fresh": false,
            "graph": output,
            "metadata": sidecar,
            "reason": "unsupported metadata format",
        }));
    }
    let source = PathBuf::from(
        metadata["source"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("metadata is missing source"))?,
    );
    let repo_root = metadata["repo_root"].as_str().map(PathBuf::from);
    let revisions: Vec<String> = metadata["revisions"]
        .as_array()
        .ok_or_else(|| anyhow::anyhow!("metadata is missing revisions"))?
        .iter()
        .filter_map(|v| v.as_str().map(str::to_string))
        .collect();
    let current = source_fingerprint(&source, repo_root.as_deref(), &revisions)?;
    let recorded = metadata["fingerprint"].as_str().unwrap_or("");
    let fresh = current == recorded;
    Ok(json!({
        "status": if fresh { "fresh" } else { "stale" },
        "fresh": fresh,
        "graph": output,
        "metadata": sidecar,
        "source": source,
        "mode": metadata["mode"],
        "revisions": revisions,
        "reason": if fresh { "source fingerprint matches" } else { "source changed since the graph was built" },
    }))
}

fn resolved_output(source: &Path, output: Option<&Path>) -> PathBuf {
    output.map(Path::to_path_buf).unwrap_or_else(|| {
        if source.is_file() {
            source.parent().unwrap_or(source).join(DEFAULT_GRAPH)
        } else {
            source.join(DEFAULT_GRAPH)
        }
    })
}

fn metadata_path(output: &Path) -> PathBuf {
    let mut path: OsString = output.as_os_str().to_os_string();
    path.push(".meta.json");
    PathBuf::from(path)
}

fn print_status(status: &Value, format: StatusFormat) {
    match format {
        StatusFormat::Json => println!("{}", serde_json::to_string(status).expect("JSON value")),
        StatusFormat::Human => {
            let state = status["status"].as_str().unwrap_or("unknown");
            let graph = status["graph"].as_str().unwrap_or("");
            println!("{state}: {graph}");
            if let Some(reason) = status["reason"].as_str() {
                println!("{reason}");
            }
        }
    }
}

fn source_fingerprint(
    source: &Path,
    repo_root: Option<&Path>,
    revisions: &[String],
) -> Result<String> {
    let mut hash = Fnv64::new();
    hash.update(source.to_string_lossy().as_bytes());
    if revisions.is_empty() {
        if source.is_file() {
            let root = source.parent().unwrap_or(source);
            hash_tree(root, root, &mut hash)?;
        } else {
            hash_tree(source, source, &mut hash)?;
        }
    } else {
        let root = repo_root.unwrap_or(source);
        for rev in revisions {
            let output = Command::new("git")
                .arg("-C")
                .arg(root)
                .args(["rev-parse", "--verify"])
                .arg(format!("{rev}^{{commit}}"))
                .output()
                .with_context(|| "failed to run git for freshness check")?;
            if !output.status.success() {
                anyhow::bail!(
                    "could not resolve git revision {:?} in {}: {}",
                    rev,
                    root.display(),
                    String::from_utf8_lossy(&output.stderr).trim()
                );
            }
            hash.update(rev.as_bytes());
            hash.update(&output.stdout);
        }
    }
    Ok(format!("fnv1a64:{:016x}", hash.finish()))
}

fn hash_tree(root: &Path, dir: &Path, hash: &mut Fnv64) -> Result<()> {
    let mut entries: Vec<_> = fs::read_dir(dir)
        .with_context(|| format!("could not read source directory {}", dir.display()))?
        .collect::<std::io::Result<Vec<_>>>()?;
    entries.sort_by_key(|entry| entry.file_name());
    for entry in entries {
        let path = entry.path();
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if entry.file_type()?.is_dir() {
            if name.starts_with('.')
                || matches!(name.as_ref(), "target" | "node_modules" | "__pycache__")
            {
                continue;
            }
            hash_tree(root, &path, hash)?;
        } else if entry.file_type()?.is_file() {
            if path.extension().is_some_and(|ext| ext == "kgl") || name.ends_with(".kgl.meta.json")
            {
                continue;
            }
            let rel = path.strip_prefix(root).unwrap_or(&path);
            hash_file(&path, rel, hash)?;
        }
    }
    Ok(())
}

fn hash_file(path: &Path, relative: &Path, hash: &mut Fnv64) -> Result<()> {
    hash.update(relative.to_string_lossy().as_bytes());
    let mut file = fs::File::open(path)?;
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hash.update(&buffer[..read]);
    }
    Ok(())
}

struct Fnv64(u64);

impl Fnv64 {
    fn new() -> Self {
        Self(0xcbf29ce484222325)
    }

    fn update(&mut self, bytes: &[u8]) {
        for byte in bytes {
            self.0 ^= u64::from(*byte);
            self.0 = self.0.wrapping_mul(0x100000001b3);
        }
    }

    fn finish(&self) -> u64 {
        self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::process::Command;

    fn git(dir: &Path, args: &[&str]) {
        let status = Command::new("git")
            .arg("-C")
            .arg(dir)
            .args(args)
            .status()
            .unwrap();
        assert!(status.success(), "git {:?} failed", args);
    }

    fn fixture() -> tempfile::TempDir {
        let tmp = tempfile::tempdir().unwrap();
        fs::write(tmp.path().join("demo.rs"), "pub fn first() {}\n").unwrap();
        git(tmp.path(), &["init", "-q"]);
        git(tmp.path(), &["config", "user.email", "test@example.com"]);
        git(tmp.path(), &["config", "user.name", "Test"]);
        git(tmp.path(), &["add", "demo.rs"]);
        git(tmp.path(), &["commit", "-qm", "first"]);
        fs::write(
            tmp.path().join("demo.rs"),
            "pub fn first() {}\npub fn second() {}\n",
        )
        .unwrap();
        git(tmp.path(), &["add", "demo.rs"]);
        git(tmp.path(), &["commit", "-qm", "second"]);
        tmp
    }

    #[test]
    fn working_tree_build_reports_fresh_then_stale() {
        let parent = tempfile::tempdir().unwrap();
        let source = parent.path().join("source with spaces");
        fs::create_dir(&source).unwrap();
        fs::write(source.join("demo.rs"), "pub fn demo() {}\n").unwrap();
        let output = source.join("code review.kgl");
        let result = build(&BuildArgs {
            source: source.clone(),
            output: Some(output.clone()),
            rev: None,
            revs: vec![],
            repo_root: None,
            no_tests: false,
            include_docs: false,
            max_loc_per_file: None,
            verbose: false,
            format: StatusFormat::Json,
        })
        .unwrap();
        assert_eq!(result["fresh"], true);
        assert_eq!(status(&output).unwrap()["fresh"], true);
        fs::write(source.join("demo.rs"), "pub fn changed() {}\n").unwrap();
        assert_eq!(status(&output).unwrap()["fresh"], false);
    }

    #[test]
    fn multi_revision_build_and_bad_revision() {
        let repo = fixture();
        let output = repo.path().join("multi.kgl");
        let args = BuildArgs {
            source: repo.path().to_path_buf(),
            output: Some(output.clone()),
            rev: None,
            revs: vec!["HEAD~1".into(), "HEAD".into()],
            repo_root: None,
            no_tests: false,
            include_docs: false,
            max_loc_per_file: None,
            verbose: false,
            format: StatusFormat::Json,
        };
        assert_eq!(build(&args).unwrap()["mode"], "multi-revision");
        assert_eq!(status(&output).unwrap()["fresh"], true);

        let bad = BuildArgs {
            revs: vec!["not-a-revision".into()],
            ..args
        };
        let error = build(&bad).unwrap_err().to_string();
        assert!(error.contains("could not resolve git revision"));
    }
}
