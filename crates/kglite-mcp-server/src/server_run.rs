//! The async server body: one named boot phase after another, ending in
//! the stdio MCP service.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result};
use mcp_methods::server::{init_tracing, load_env_for_mode, McpServer, ServerOptions};
use rmcp::transport::stdio;
use rmcp::ServiceExt;

use crate::tools::GraphState;
use crate::*;

/// Compile and validate the manifest's stored Cypher recipes once at boot.
///
/// Recipe definitions are configuration, not per-call input, so the
/// deliberately small schema subset is checked here and later route handlers
/// share one immutable catalog.
fn boot_recipe_catalog(
    manifest: Option<&mcp_methods::server::Manifest>,
) -> Result<Arc<recipe_queries::RecipeCatalog>> {
    let catalog = Arc::new(match manifest {
        Some(manifest) => recipe_queries::RecipeCatalog::from_manifest_value(
            manifest.extensions.get("cypher_recipes"),
        )
        .context("extensions.cypher_recipes parse failed")?,
        None => recipe_queries::RecipeCatalog::default(),
    });
    if let Some(summary) = catalog.discovery_summary() {
        tracing::info!(
            recipes = summary.recipe_count,
            queries = summary.query_count,
            "Cypher recipe catalog validated"
        );
    }
    Ok(catalog)
}

/// Build the manifest-declared, position-scoped literal codecs
/// (`extensions.value_codecs`: prefix / map / regex).
///
/// Passed to the engine via `ExecuteOptions` per query — decode on the way in,
/// encode on the way out. Empty when absent; a malformed block errors at boot,
/// not per-query.
fn boot_value_codecs(
    manifest: Option<&mcp_methods::server::Manifest>,
) -> Result<Option<Arc<Vec<kglite::api::cypher::ValueCodec>>>> {
    let codecs = match manifest {
        Some(m) => value_codecs::from_manifest(m.extensions.get("value_codecs"))
            .context("extensions.value_codecs parse failed")?,
        None => Vec::new(),
    };
    Ok((!codecs.is_empty()).then(|| Arc::new(codecs)))
}

/// `extensions.graph_watch` — **retired**, and still parsed.
///
/// What it used to arm is now unconditional: a `--graph` server stats the
/// served file on every tool call (`tools::graph_reload`), so there is no
/// watcher to switch on and nothing an operator gains by setting the key.
/// Returning what was parsed rather than dropping it is what lets the caller
/// warn about a key that no longer does anything — a manifest carrying
/// `graph_watch: false` used to *mean* "do not refresh", and its author has to
/// learn that it no longer says that. A non-boolean value is still a boot
/// error rather than a silently ignored key (see [`boot_tools_allow`]): the
/// day this key is finally deleted, a typo in it must not start passing.
/// `extensions.ontology: {"file": "x.json"}` — the declared semantic layer,
/// parsed once at boot (a malformed value fails the server, per the crate's
/// no-silently-ignored-keys rule) and applied memory-only to every graph the
/// state installs. The path resolves against the manifest's directory, like
/// `csv_http_server`.
use crate::tools::BoundOntology;

fn boot_ontology(
    manifest: Option<&mcp_methods::server::Manifest>,
    manifest_base: &Path,
) -> Result<Option<BoundOntology>> {
    let Some(raw) = manifest.and_then(|m| m.extensions.get("ontology")) else {
        return Ok(None);
    };
    let materialize = match raw.get("materialize") {
        None => false,
        Some(v) => v
            .as_bool()
            .context("extensions.ontology.materialize must be a boolean")?,
    };
    let file = raw
        .get("file")
        .and_then(|v| v.as_str())
        .context("extensions.ontology must be a map with a 'file' path (JSON document)")?;
    let path = if Path::new(file).is_absolute() {
        PathBuf::from(file)
    } else {
        manifest_base.join(file)
    };
    let text = std::fs::read_to_string(&path)
        .with_context(|| format!("extensions.ontology: cannot read {}", path.display()))?;
    let store = kglite::api::ontology_from_json(&text)
        .map_err(|e| anyhow::anyhow!("extensions.ontology {}: {e}", path.display()))?;
    tracing::info!(
        classes = store.classes.len(),
        relationships = store.relationships.len(),
        materialize,
        "manifest ontology parsed (memory-only; persists only via an explicit save_graph)"
    );
    Ok(Some(BoundOntology {
        store: Arc::new(store),
        materialize,
    }))
}

fn boot_graph_watch(manifest: Option<&mcp_methods::server::Manifest>) -> Result<Option<bool>> {
    let Some(raw) = manifest.and_then(|m| m.extensions.get("graph_watch")) else {
        return Ok(None);
    };
    raw.as_bool()
        .map(Some)
        .context("extensions.graph_watch must be a boolean (true or false)")
}

/// `extensions.parallel: true` — the manifest half of the parallel opt-in.
///
/// Boolean-key handling matches [`boot_tools_allow`]: absent is off, a
/// non-boolean is a boot error rather than a silently dropped key.
fn boot_manifest_parallel(manifest: Option<&mcp_methods::server::Manifest>) -> Result<bool> {
    let Some(raw) = manifest.and_then(|m| m.extensions.get("parallel")) else {
        return Ok(false);
    };
    raw.as_bool()
        .context("extensions.parallel must be a boolean (true or false)")
}

/// Resolve the parallel-runtime opt-in from `--parallel` and
/// `extensions.parallel`.
///
/// **OR, not intersection** — the opposite of [`boot_write_scope`], and for
/// the reason that makes the two opposite: a write scope is a *perimeter*, so
/// combining two of them may only narrow, while this is a *resource
/// permission* whose worst case is a query using more cores than it needed.
/// Either surface alone turns it on, so a wrapper that owns the manifest but
/// not argv, and a bare binary launched with no manifest at all, each have a
/// working way to say yes.
///
/// Applies to the read seam only ([`tools::ExecPolicy`]); mutations stay
/// sequential.
fn boot_parallel(manifest: Option<&mcp_methods::server::Manifest>, cli: &Cli) -> Result<bool> {
    Ok(boot_manifest_parallel(manifest)? || cli.parallel)
}

/// Fallback name for the lease record: the process that spawned this server.
///
/// Best-effort by construction — every failure returns `None` and the caller
/// falls back to the binary's own name. Zero new dependencies: `/proc` on
/// Linux, one `ps` on macOS, nothing on Windows (where the owner record still
/// carries pid and timestamp).
#[cfg(unix)]
fn parent_process_name() -> Option<String> {
    let ppid = std::os::unix::process::parent_id();
    #[cfg(target_os = "linux")]
    let raw = std::fs::read_to_string(format!("/proc/{ppid}/comm")).ok()?;
    #[cfg(not(target_os = "linux"))]
    let raw = {
        let output = std::process::Command::new("ps")
            .args(["-o", "comm=", "-p", &ppid.to_string()])
            .output()
            .ok()?;
        if !output.status.success() {
            return None;
        }
        String::from_utf8(output.stdout).ok()?
    };
    // macOS `ps -o comm=` reports the executable path, not the basename.
    let name = raw.trim().rsplit('/').next()?.trim();
    (!name.is_empty()).then(|| name.to_owned())
}

#[cfg(not(unix))]
fn parent_process_name() -> Option<String> {
    None
}

/// Resolve the name this server publishes in the graph's `<path>.lock-owner`
/// record: `--lease-label`, else `KGLITE_LEASE_LABEL`, else the parent process
/// name, else the binary's own name.
///
/// Mirrors [`boot_parallel`]'s shape — resolved once at boot, carried by every
/// clone of the state. There is deliberately **no manifest key**: the case this
/// exists for is four MCP clients sharing one manifest on one graph, so a
/// manifest-declared label would give all four the same name and answer
/// nothing. The env var is read explicitly rather than through clap's `env`
/// attribute so the precedence above is visible here rather than in an
/// attribute two files away.
fn boot_lease_label(cli: &Cli) -> String {
    if let Some(label) = cli.lease_label.as_deref().map(str::trim) {
        if !label.is_empty() {
            return label.to_owned();
        }
    }
    if let Ok(label) = std::env::var("KGLITE_LEASE_LABEL") {
        let label = label.trim();
        if !label.is_empty() {
            return label.to_owned();
        }
    }
    parent_process_name().unwrap_or_else(|| "kglite-mcp-server".to_owned())
}

/// The width the engine's query pool will be built at, for the boot log.
///
/// Recomputed here rather than read from the engine: the pool is built lazily
/// on the first query that crosses a fan-out threshold, so at boot there is
/// nothing to ask. Mirrors `kglite::graph::parallel::configured_width` — same
/// env var, same `available_parallelism` fallback — and is reported as
/// observed configuration, never as a promise about a pool that does not
/// exist yet.
fn query_pool_width() -> (usize, bool) {
    let override_width = std::env::var_os("KGLITE_QUERY_THREADS")
        .and_then(|raw| raw.to_str()?.trim().parse::<usize>().ok())
        .filter(|n| *n > 0);
    match override_width {
        Some(n) => (n, true),
        None => (
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1),
            false,
        ),
    }
}

/// `extensions.tools_allow: [name, ...]` — the closed-by-default tool surface.
///
/// Absent key = no allowlist = every registered route stays visible. Present,
/// it names the *final* tool surface: see [`apply_tool_allowlist`] for what
/// that means and what it deliberately does not do. An explicit empty list is
/// honoured literally — a server with no tools — because the alternative
/// (silently treating `[]` as "no allowlist") would widen a surface an
/// operator asked to close.
///
/// A non-list value, or a non-string element, is a boot error rather than a
/// dropped key: an allowlist that silently fails open is worse than no
/// allowlist, and "parsed and then ignored" is this crate's recurring defect
/// shape (`temp_cleanup`).
fn boot_tools_allow(
    manifest: Option<&mcp_methods::server::Manifest>,
) -> Result<Option<Vec<String>>> {
    let Some(raw) = manifest.and_then(|m| m.extensions.get("tools_allow")) else {
        return Ok(None);
    };
    let items = raw
        .as_array()
        .context("extensions.tools_allow must be a list of tool names")?;
    let mut names = Vec::with_capacity(items.len());
    for item in items {
        let name = item.as_str().with_context(|| {
            format!("extensions.tools_allow entries must be tool names (strings); found {item}")
        })?;
        names.push(name.to_owned());
    }
    Ok(Some(names))
}

/// `extensions.write_scope: [NodeType, ...]` — the operator-pinned write scope.
///
/// Absent key = no pin = the agent's own `write_scope` argument decides (which
/// may be nothing at all). Present, it is the ceiling the agent can only narrow:
/// see [`resolve_write_scope`] for how the two combine. An explicit empty list
/// is honoured literally — a write-enabled server that permits no writes —
/// because reading `[]` as "no pin" would widen a perimeter the operator asked
/// to close.
///
/// A non-list value, or a non-string element, is a boot error rather than a
/// dropped key, for the same reason [`boot_tools_allow`] fails that way.
fn boot_manifest_write_scope(
    manifest: Option<&mcp_methods::server::Manifest>,
) -> Result<Option<Vec<String>>> {
    let Some(raw) = manifest.and_then(|m| m.extensions.get("write_scope")) else {
        return Ok(None);
    };
    let items = raw
        .as_array()
        .context("extensions.write_scope must be a list of node types")?;
    let mut names = Vec::with_capacity(items.len());
    for item in items {
        let name = item.as_str().with_context(|| {
            format!("extensions.write_scope entries must be node types (strings); found {item}")
        })?;
        names.push(name.to_owned());
    }
    Ok(Some(names))
}

/// Resolve the operator-pinned write scope from the manifest key and the
/// `--write-scope` flag.
///
/// Both set is not an error: the two are **intersected**, the same rule the
/// agent's own scope obeys, so neither surface can widen what the other pinned.
/// Refusing the ambiguity at boot would only push the operator into deleting
/// one of them. The intersection is logged, so the effective scope shows in the
/// boot output rather than having to be inferred.
fn boot_write_scope(
    manifest: Option<&mcp_methods::server::Manifest>,
    cli: &Cli,
) -> Result<Option<Vec<String>>> {
    let from_manifest = boot_manifest_write_scope(manifest)?;
    let from_flag = cli.write_scope.as_deref().map(parse_write_scope_flag);
    let resolved = match (from_manifest, from_flag) {
        (None, None) => None,
        (Some(only), None) | (None, Some(only)) => Some(only),
        (Some(m), Some(f)) => {
            let both: Vec<String> = f.iter().filter(|t| m.contains(t)).cloned().collect();
            tracing::info!(
                manifest = m.join(","),
                flag = f.join(","),
                effective = both.join(","),
                "write_scope pinned by both --write-scope and extensions.write_scope; \
                 the effective scope is their intersection"
            );
            Some(both)
        }
    };
    // `cli.writable`, not `graph_writes_enabled`: a manifest that only enables
    // `builtins.save_graph` registers `save_graph` but still serves the
    // read-only `cypher_query`, so the pin has nothing to apply to there
    // either.
    if resolved.is_some() && !cli.writable {
        tracing::warn!(
            "--write-scope / extensions.write_scope is set on a server whose cypher_query is \
             read-only; every mutation is refused already. Add --writable to serve scoped writes."
        );
    }
    Ok(resolved)
}

/// Directory the manifest was loaded from — the base both `csv_http_server`
/// (resolving `dir:`) and `temp_cleanup` (finding the directory to wipe)
/// resolve against. Falls back to cwd when there's no manifest.
fn manifest_base_dir(manifest: Option<&mcp_methods::server::Manifest>) -> PathBuf {
    manifest
        .and_then(|m| m.yaml_path.parent().map(|p| p.to_path_buf()))
        .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")))
}

/// Start the opt-in `extensions.csv_http_server:` CSV-over-HTTP listener.
///
/// When configured we spawn a tokio task to serve files out of the directory;
/// the `cypher_query` tool sees the same state and writes `FORMAT CSV` results
/// to that directory, returning a URL instead of an inline CSV blob.
///
/// `Err` is a malformed `csv_http_server` value only — a config-syntax error,
/// fatal like every other manifest mistake. A well-formed config whose
/// listener fails to *start* comes back as [`csv_http::CsvHttpState::Failed`]:
/// the extension is off, the boot continues, and both the summary line and
/// every `FORMAT CSV` answer say why.
async fn boot_csv_http(
    manifest: Option<&mcp_methods::server::Manifest>,
    manifest_base: &Path,
) -> Result<csv_http::CsvHttpState> {
    let cfg = match manifest.and_then(|m| m.extensions.get("csv_http_server")) {
        Some(raw) => csv_http::CsvHttpConfig::from_manifest_value(raw, manifest_base)
            .context("extensions.csv_http_server parse failed")?,
        None => None,
    };
    Ok(match cfg {
        Some(cfg) => csv_http::spawn(cfg).await,
        None => csv_http::CsvHttpState::Off,
    })
}

/// Whether this deployment can write the graph file it serves — `--writable`
/// (which implies `save_graph`: an agent that mutates needs to persist) or a
/// manifest that enables `save_graph` on its own.
///
/// Read twice, once for each thing that follows from it: which write tools get
/// registered ([`tools::Builtins`]) and whether the server takes the path's
/// cross-process writer lease ([`writer_lease_policy`]). One function so those
/// two can never disagree about what "this server writes the graph" means.
fn graph_writes_enabled(manifest: Option<&mcp_methods::server::Manifest>, cli: &Cli) -> bool {
    manifest.map(|m| m.builtins.save_graph).unwrap_or(false) || cli.writable
}

/// Whether to take the single-writer lease on the served path.
///
/// A read-only server never rewrites the file, so holding its lease refuses
/// every external rebuilder (`kglite.open(path)` fails fast) for the server's
/// whole lifetime and buys nothing back — and it stops two read-only servers
/// from serving one `.kgl`. Write-enabled deployments keep the exclusive lease;
/// so do disk-graph directories, whose retained mmaps make an external writer
/// genuinely unsafe rather than merely stale (see
/// `GraphState::takes_writer_lease`).
fn writer_lease_policy(
    manifest: Option<&mcp_methods::server::Manifest>,
    cli: &Cli,
) -> tools::WriterLeasePolicy {
    if graph_writes_enabled(manifest, cli) {
        tools::WriterLeasePolicy::Exclusive
    } else {
        tools::WriterLeasePolicy::ReadOnly
    }
}

/// Builtin toggles from the manifest.
///   - `save_graph`: registration gated on [`graph_writes_enabled`], so a
///     destructive operation is not exposed to the agent on every graph
///     regardless of intent.
///   - `temp_cleanup: on_overview`: wipe `temp/` on every bare
///     `graph_overview()`.
fn boot_builtins(
    manifest: Option<&mcp_methods::server::Manifest>,
    cli: &Cli,
    csv_http: &csv_http::CsvHttpState,
    manifest_base: &Path,
    write_scope: Option<Vec<String>>,
) -> tools::Builtins {
    tools::Builtins {
        save_graph: graph_writes_enabled(manifest, cli),
        writable: cli.writable,
        write_scope,
        temp_cleanup_on_overview: manifest
            .map(|m| {
                matches!(
                    m.builtins.temp_cleanup,
                    mcp_methods::server::manifest::TempCleanup::OnOverview
                )
            })
            .unwrap_or(false),
        // Resolved against the manifest base, not the cwd — the server's cwd
        // need not match the manifest's parent. Reuse the csv_http_server
        // directory when configured, so both sides of the CSV pipeline agree
        // on what counts as "the temp dir".
        temp_dir: Some(
            csv_http
                .dir()
                .map(Path::to_path_buf)
                .unwrap_or_else(|| manifest_base.join("temp")),
        ),
    }
}

fn register_kglite_tools(
    server: &mut McpServer,
    graph_state: &GraphState,
    manifest: Option<&mcp_methods::server::Manifest>,
    builtins: tools::Builtins,
    recipe_catalog_summary: Option<crate::recipe_queries::CatalogSummary>,
    csv_http: Arc<csv_http::CsvHttpState>,
    source_roots_provider: Option<mcp_methods::server::source::SourceRootsProvider>,
) -> Result<()> {
    tools::register(
        server,
        graph_state.clone(),
        builtins,
        tools::OverviewDecorations {
            prefix: manifest.and_then(|manifest| manifest.overview_prefix.clone()),
            catalog: recipe_catalog_summary,
        },
        csv_http,
    );
    code_source::register(server, graph_state.clone(), source_roots_provider.clone())
        .context("read_code_source registration failed")?;
    explore::register(server, graph_state.clone(), source_roots_provider)
        .context("explore registration failed")?;
    Ok(())
}

/// Whether `explore` and `read_code_source` should be hidden for this boot.
///
/// `explore` pins its entry types and its traversal edge whitelist to code node
/// types, so on a graph without `Function`/`Class` it can only ever return "no
/// match" — it is structurally dead, and its presence in `tools/list` is pure
/// misdirection. Worse, `read_code_source`'s optional `node_type` argument makes
/// it a general-purpose reader of whatever `file_path` properties the graph
/// happens to carry, so on a non-code graph it is a way to pull arbitrary files
/// off disk with no code-graph purpose to justify it.
///
/// Three constraints are deliberate and settled:
///
/// 1. **This is a boot-time snapshot of a runtime property.** Every other
///    `has_node_type` consumer (skill `applies_when:` predicates, the result
///    footers) re-evaluates per call; this one cannot, because tool handlers
///    have no access to the router and so nothing can re-enable a route once
///    the server is running. A same-path graph reload rarely changes the
///    graph's *class* (a code graph stays a code graph), so the staleness
///    window is accepted. If handlers ever gain router access, make it live.
/// 2. **Writable servers are exempt.** `load_graph` can swap a code graph in
///    at any time, and by (1) there would be no way to re-enable the routes
///    afterwards — so a writable server keeps both tools regardless of what
///    it booted with.
/// 3. **Graph mode only.** Every other mode has no graph at boot, so
///    `has_node_type` is uniformly `false` there and a mode-blind gate would
///    strip `explore` from exactly the deployments that need it most
///    (local-workspace and GitHub-repo code servers, whose graphs are built
///    after registration).
fn code_tools_are_dead(mode: &Mode, builtins: &tools::Builtins, graph_state: &GraphState) -> bool {
    matches!(mode, Mode::Graph { .. })
        && !builtins.writable
        && !(graph_state.has_node_type("Function") || graph_state.has_node_type("Class"))
}

/// Bind the manifest's `extensions.embedder:` as the embedder `text_score()`
/// uses.
///
/// Selection is by `library:` — `fastembed-rs` for the Rust-native adapter (the
/// only option on the standalone libpython-free binary), anything else for a
/// Python library built by the wheel-supplied factory. The rules, and the
/// `trust.allow_embedder` gate they sit behind, live in
/// [`build_embedder_from_manifest`].
fn bind_manifest_embedder(
    manifest: Option<&mcp_methods::server::Manifest>,
    py_embedder_factory: Option<&PyEmbedderFactory>,
    graph_state: &GraphState,
) -> Result<()> {
    if let Some(m) = manifest {
        if let Some(embedder) = build_embedder_from_manifest(m, py_embedder_factory)? {
            graph_state
                .bind_embedder(embedder)
                .context("graph.set_embedder_native failed")?;
        }
    }
    Ok(())
}

pub(crate) async fn run_async(
    cli: Cli,
    py_embedder_factory: Option<PyEmbedderFactory>,
    extensions: ServerExtensions,
) -> Result<()> {
    let ServerExtensions {
        workspace_graph,
        domain_tools,
    } = extensions;
    init_tracing();
    let mode = pick_mode(&cli);
    validate_mode_paths(&mode, &cli)?;

    let manifest = load_manifest(&cli, &mode).context("manifest load failed")?;
    let recipe_catalog = boot_recipe_catalog(manifest.as_ref())?;
    let recipe_catalog_summary = recipe_catalog.discovery_summary();
    // Parsed here, at the top of boot, so a malformed value fails the server
    // rather than surfacing as a key that quietly does nothing.
    if boot_graph_watch(manifest.as_ref())?.is_some() {
        tracing::warn!(
            "extensions.graph_watch is retired — a --graph server now refreshes automatically \
             on every tool call; remove the key"
        );
    }
    let tools_allow = boot_tools_allow(manifest.as_ref())?;
    let write_scope = boot_write_scope(manifest.as_ref(), &cli)?;
    let parallel = boot_parallel(manifest.as_ref(), &cli)?;
    if parallel {
        let (width, overridden) = query_pool_width();
        tracing::info!(
            pool_threads = width,
            width_source = if overridden {
                "KGLITE_QUERY_THREADS"
            } else {
                "available_parallelism"
            },
            "parallel runtime enabled for reads (--parallel / extensions.parallel); \
             mutations stay sequential"
        );
    }

    // Manifest `workspace.kind: local` wins over CLI flags — promote before
    // mode-specific binding so the rest of boot sees `Mode::LocalWorkspace`.
    let mode = promote_local_workspace(mode, manifest.as_ref())?;

    // Load `.env` before anything reads env vars (notably the GitHub tools'
    // `GITHUB_TOKEN` auth check). The walk-up start point matches the framework
    // binary's choice in `mcp-server`'s own main: the mode's directory for
    // source-aware modes, cwd for bare. An explicit manifest `env_file:`
    // overrides walk-up. Returns the path actually loaded, so the boot summary
    // can name it.
    let env_start_dir = resolve_env_start_dir(&mode);
    let env_file_loaded = load_env_for_mode(manifest.as_ref(), &env_start_dir)
        .context("manifest env_file load failed")?;

    let mut options = ServerOptions::from_manifest(manifest.as_ref(), fallback_name(&mode));
    if cli.name.is_some() {
        options.name = cli.name.clone();
    }
    // Fold the lazy-tool-discovery steer into workspace-mode instructions so
    // code-mode / tool-search clients get the "search the registry for cypher"
    // guidance by default, without every deployment copy-pasting it.
    let options = apply_discovery_steer(&mode, options);

    let graph_state = GraphState::new(workspace_graph_mode(&mode))
        .with_value_codecs(boot_value_codecs(manifest.as_ref())?)
        .with_parallel(parallel)
        .with_workspace_graph(workspace_graph.map(Arc::new))
        // Declared here rather than from `builtins` (built below, after the
        // csv_http boot it needs) because `bind_mode` performs the boot open on
        // the very next line — a policy set later would arrive after the lease
        // decision it governs. Both read the same `graph_writes_enabled`.
        .with_writer_lease_policy(writer_lease_policy(manifest.as_ref(), &cli))
        .with_lease_label(Some(boot_lease_label(&cli)));

    // Bound before `bind_mode`, whose boot open publishes the first graph —
    // the ontology rides the same pre-publication seam as the embedder.
    if let Some(bound) = boot_ontology(manifest.as_ref(), &manifest_base_dir(manifest.as_ref()))? {
        graph_state.bind_ontology(bound);
    }

    let (options, source_root_status) =
        bind_mode(&mode, &cli, manifest.as_ref(), &graph_state, options)?;

    let options = apply_result_decorations(options, &graph_state, source_root_status.as_ref());

    // Snapshot the dynamic source-roots provider before `options` moves into
    // the McpServer. `read_code_source` queries it on every call, so
    // workspace-mode active-repo swaps immediately re-target file resolution.
    let source_roots_provider = options.source_roots.clone();

    let manifest_base = manifest_base_dir(manifest.as_ref());
    let csv_http = Arc::new(boot_csv_http(manifest.as_ref(), &manifest_base).await?);
    let builtins = boot_builtins(
        manifest.as_ref(),
        &cli,
        &csv_http,
        &manifest_base,
        write_scope,
    );

    let mut server = McpServer::new(options);
    // `repo_management` needs no gate here since mcp-methods 0.4.7:
    // `McpServer::new` registers it only for `kind: github` workspaces, so
    // local-workspace (and graph/bare) servers never carry the route. The
    // manifest-override consequence of that removal is handled where the
    // overrides are applied — see the `repo_management` pass in
    // `apply_bundled_tool_overrides`.
    //
    // Read before `builtins` moves into `register_kglite_tools` below.
    let gate_code_tools = code_tools_are_dead(&mode, &builtins, &graph_state);
    register_kglite_tools(
        &mut server,
        &graph_state,
        manifest.as_ref(),
        builtins,
        recipe_catalog_summary,
        csv_http.clone(),
        source_roots_provider,
    )?;
    if matches!(mode, Mode::Graph { .. }) {
        // `reload_graph` is graph-mode-only: it re-reads *the* served file, an
        // identity no other mode has. Registered from here rather than inside
        // `tools::register`, because the mode is not otherwise visible there.
        tools::register_graph_mode_tools(&mut server, graph_state.clone());
    }
    if gate_code_tools {
        // Disable, never skip registration: an unregistered name is absent from
        // `router.map`, and `apply_bundled_tool_overrides` hard-errors on any
        // manifest override naming a route not in that map (only the
        // framework-gated `repo_management` gets a pass there). `disable_route`
        // keeps the entry — unlisted, rejected on call — while leaving
        // overrides resolvable.
        server.tool_router_mut().disable_route("explore");
        server.tool_router_mut().disable_route("read_code_source");
    }
    bind_manifest_embedder(
        manifest.as_ref(),
        py_embedder_factory.as_ref(),
        &graph_state,
    )?;

    // Register YAML Cypher tools, then downstream domain routes. Keeping this
    // before skill finalisation makes every route visible to predicates.
    register_extension_tools(
        &mut server,
        &graph_state,
        manifest.as_ref(),
        &csv_http,
        recipe_catalog,
        domain_tools,
    )?;

    // Last, and only here: the allowlist closes the surface once *every* route
    // source has registered and `apply_bundled_tool_overrides` (at the tail of
    // `register_extension_tools`) has settled the final names. Applying it any
    // earlier would let a later registration re-widen the surface, and would
    // match against pre-rename names the agent never sees. Before skills, so a
    // `tool_registered:` predicate sees the closed surface.
    if let Some(allow) = tools_allow.as_deref() {
        apply_tool_allowlist(&mut server, allow, recipe_catalog_summary.is_some())
            .context("extensions.tools_allow could not be applied")?;
    }

    let _watch_handle = spawn_mode_watcher(&mode, &graph_state)?;

    // Bare-mode (no manifest) deployments get no skills — the `skills:`
    // declaration lives in the manifest.
    if let Some(m) = manifest.as_ref() {
        install_skills(&mut server, m, &graph_state, recipe_catalog_summary);
    }

    print_boot_summary(
        &mode,
        manifest.as_ref(),
        &graph_state,
        env_file_loaded.as_deref(),
        &csv_http,
        source_root_status.as_ref(),
    );

    let service = server
        .serve(stdio())
        .await
        .context("failed to start MCP service over stdio")?;
    service.waiting().await?;
    Ok(())
}

/// Every boot knob read here must survive the whole chain: spelled in a YAML
/// file, accepted by mcp-methods' loader (which validates nothing inside
/// `extensions:`), and read back here as the type the wiring expects. Loading
/// through the real loader rather than constructing a `Manifest` is deliberate
/// — a struct literal would skip the two steps most likely to break.
#[cfg(test)]
mod boot_manifest_tests {
    use super::*;

    fn manifest_with(dir: &Path, body: &str) -> mcp_methods::server::Manifest {
        let path = dir.join("graph_watch_mcp.yaml");
        std::fs::write(&path, body).expect("write manifest");
        mcp_methods::server::load_manifest(&path).expect("manifest loads")
    }

    /// The retired key is *accepted*, not honoured: a manifest that still
    /// carries it must boot (four MCP clients share one manifest, and an
    /// operator cannot edit it the moment this server upgrades), and the boot
    /// must be able to tell it apart from an absent key so it can say the key
    /// no longer does anything. Both values are declared present, including
    /// `false` — that is the one an author wrote to mean "never refresh", and
    /// it is the one whose meaning this release took away.
    #[test]
    fn a_graph_watch_key_is_accepted_and_reported_as_retired() {
        let tmp = tempfile::tempdir().expect("tempdir");
        assert_eq!(
            boot_graph_watch(None).expect("no manifest parses"),
            None,
            "no manifest carries no key to warn about"
        );
        let bare = manifest_with(tmp.path(), "name: bare\n");
        assert_eq!(
            boot_graph_watch(Some(&bare)).expect("bare manifest parses"),
            None,
            "an absent key must not produce a retirement warning"
        );
        let off = manifest_with(tmp.path(), "name: off\nextensions:\n  graph_watch: false\n");
        assert_eq!(
            boot_graph_watch(Some(&off)).expect("explicit false parses"),
            Some(false),
            "`false` no longer disables anything, so its author must be told"
        );
        let on = manifest_with(tmp.path(), "name: on\nextensions:\n  graph_watch: true\n");
        assert_eq!(
            boot_graph_watch(Some(&on)).expect("explicit true parses"),
            Some(true),
            "and `true` is now what every --graph server does anyway"
        );
    }

    /// The name a peer reads instead of "another process". Precedence is
    /// `--lease-label` > `KGLITE_LEASE_LABEL` > parent process name; only the
    /// flag half is asserted here, because the other two read process state a
    /// unit test cannot set without racing every other test in the binary.
    #[test]
    fn an_explicit_lease_label_wins_and_a_blank_one_falls_through() {
        let named = Cli::parse_from([
            "kglite-mcp-server",
            "--graph",
            "g.kgl",
            "--lease-label",
            "  Claude Desktop  ",
        ]);
        assert_eq!(boot_lease_label(&named), "Claude Desktop");
        // A blank flag is not a name. Falling through matters because the
        // fallback is what a peer reads in the owner record instead of
        // "another process".
        let blank = Cli::parse_from([
            "kglite-mcp-server",
            "--graph",
            "g.kgl",
            "--lease-label",
            " ",
        ]);
        assert!(!boot_lease_label(&blank).is_empty());
    }

    /// Boot wiring for the writer-lease policy: which deployments own the
    /// served path. The state-level consequences (lease taken or not, disk
    /// directories exempt) are pinned in `tools::tests::lifecycle`; this pins
    /// only the mapping from CLI/manifest to policy.
    #[test]
    fn only_write_enabled_deployments_take_the_writer_lease() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let read_only = Cli::parse_from(["kglite-mcp-server", "--graph", "g.kgl"]);
        assert_eq!(
            writer_lease_policy(None, &read_only),
            tools::WriterLeasePolicy::ReadOnly,
            "a read-only --graph server must leave the file lockable by a rebuilder"
        );

        let writable = Cli::parse_from(["kglite-mcp-server", "--graph", "g.kgl", "--writable"]);
        assert_eq!(
            writer_lease_policy(None, &writable),
            tools::WriterLeasePolicy::Exclusive,
            "--writable owns the file it serves"
        );

        // `save_graph` without `--writable`: still a writer of that file.
        let saver = manifest_with(tmp.path(), "name: saver\nbuiltins:\n  save_graph: true\n");
        assert_eq!(
            writer_lease_policy(Some(&saver), &read_only),
            tools::WriterLeasePolicy::Exclusive
        );
        let plain = manifest_with(tmp.path(), "name: plain\n");
        assert_eq!(
            writer_lease_policy(Some(&plain), &read_only),
            tools::WriterLeasePolicy::ReadOnly,
            "a manifest that enables nothing must not re-arm the lease"
        );
    }

    /// The absent-key arm is what makes the present-key arm mean anything.
    #[test]
    fn tools_allow_defaults_to_absent_and_reads_the_manifest_list() {
        let tmp = tempfile::tempdir().expect("tempdir");
        assert!(boot_tools_allow(None)
            .expect("no manifest parses")
            .is_none());
        let bare = manifest_with(tmp.path(), "name: bare\n");
        assert!(
            boot_tools_allow(Some(&bare))
                .expect("bare manifest parses")
                .is_none(),
            "an absent key must leave the surface open"
        );

        let listed = manifest_with(
            tmp.path(),
            "name: listed\nextensions:\n  tools_allow:\n    - cypher_query\n    - graph_overview\n    - ping\n",
        );
        assert_eq!(
            boot_tools_allow(Some(&listed)).expect("list parses"),
            Some(vec![
                "cypher_query".to_string(),
                "graph_overview".to_string(),
                "ping".to_string(),
            ]),
            "extensions.tools_allow must reach the router wiring; if this fails the key is \
             parsed and then dropped"
        );

        let empty = manifest_with(tmp.path(), "name: empty\nextensions:\n  tools_allow: []\n");
        assert_eq!(
            boot_tools_allow(Some(&empty)).expect("empty list parses"),
            Some(Vec::new()),
            "an explicit empty list closes the surface completely — it is not 'no allowlist'"
        );
    }

    #[test]
    fn a_malformed_tools_allow_fails_boot() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let scalar = manifest_with(
            tmp.path(),
            "name: scalar\nextensions:\n  tools_allow: cypher_query\n",
        );
        let error = boot_tools_allow(Some(&scalar))
            .expect_err("a scalar must fail boot, not be ignored")
            .to_string();
        assert!(error.contains("must be a list"), "{error}");

        let null = manifest_with(tmp.path(), "name: null\nextensions:\n  tools_allow:\n");
        assert!(
            boot_tools_allow(Some(&null)).is_err(),
            "an empty (null) value is a typo, not an empty list"
        );

        let mistyped = manifest_with(
            tmp.path(),
            "name: mistyped\nextensions:\n  tools_allow:\n    - ping\n    - 7\n",
        );
        let error = boot_tools_allow(Some(&mistyped))
            .expect_err("a non-string element must fail boot")
            .to_string();
        assert!(error.contains("must be tool names"), "{error}");
    }

    #[test]
    fn write_scope_reads_the_flag_the_manifest_and_their_intersection() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let bare_cli = Cli::parse_from(["kglite-mcp-server", "--graph", "g.kgl", "--writable"]);
        assert!(
            boot_write_scope(None, &bare_cli)
                .expect("no pin parses")
                .is_none(),
            "no flag and no manifest must leave writes unpinned"
        );

        let flag = Cli::parse_from([
            "kglite-mcp-server",
            "--graph",
            "g.kgl",
            "--writable",
            "--write-scope",
            "Plan,Task",
        ]);
        assert_eq!(
            boot_write_scope(None, &flag).expect("flag parses"),
            Some(vec!["Plan".to_string(), "Task".to_string()])
        );

        let listed = manifest_with(
            tmp.path(),
            "name: listed\nextensions:\n  write_scope:\n    - Plan\n    - Task\n",
        );
        assert_eq!(
            boot_write_scope(Some(&listed), &bare_cli).expect("list parses"),
            Some(vec!["Plan".to_string(), "Task".to_string()]),
            "extensions.write_scope must reach the tool wiring; if this fails the key is \
             parsed and then dropped"
        );

        // Both surfaces set: neither can widen the other.
        let narrower = Cli::parse_from([
            "kglite-mcp-server",
            "--graph",
            "g.kgl",
            "--writable",
            "--write-scope",
            "Task,Algorithm",
        ]);
        assert_eq!(
            boot_write_scope(Some(&listed), &narrower).expect("both parse"),
            Some(vec!["Task".to_string()]),
            "flag + manifest is their intersection, not either one alone"
        );

        let empty = manifest_with(tmp.path(), "name: empty\nextensions:\n  write_scope: []\n");
        assert_eq!(
            boot_write_scope(Some(&empty), &bare_cli).expect("empty list parses"),
            Some(Vec::new()),
            "an explicit empty list pins 'no writes' — it is not 'no pin'"
        );
    }

    #[test]
    fn a_malformed_write_scope_fails_boot() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let cli = Cli::parse_from(["kglite-mcp-server", "--graph", "g.kgl", "--writable"]);

        let scalar = manifest_with(
            tmp.path(),
            "name: scalar\nextensions:\n  write_scope: Task\n",
        );
        let error = boot_write_scope(Some(&scalar), &cli)
            .expect_err("a scalar must fail boot, not be ignored")
            .to_string();
        assert!(error.contains("must be a list"), "{error}");

        let null = manifest_with(tmp.path(), "name: null\nextensions:\n  write_scope:\n");
        assert!(
            boot_write_scope(Some(&null), &cli).is_err(),
            "an empty (null) value is a typo, not an empty list"
        );

        let mistyped = manifest_with(
            tmp.path(),
            "name: mistyped\nextensions:\n  write_scope:\n    - Task\n    - 7\n",
        );
        let error = boot_write_scope(Some(&mistyped), &cli)
            .expect_err("a non-string element must fail boot")
            .to_string();
        assert!(error.contains("must be node types"), "{error}");
    }

    #[test]
    fn a_non_boolean_graph_watch_fails_boot() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let bad = manifest_with(
            tmp.path(),
            "name: bad\nextensions:\n  graph_watch: yes-please\n",
        );
        let error = boot_graph_watch(Some(&bad))
            .expect_err("a non-boolean value must fail boot, not be ignored")
            .to_string();
        assert!(error.contains("must be a boolean"), "{error}");
    }

    /// The parallel opt-in is an operator *pair* — `--parallel` and
    /// `extensions.parallel` — so the boot resolution is the OR of the two.
    /// Pinned here because a manifest-only deployment (a wrapper that never
    /// controls argv) and a flag-only one (the bare binary, no manifest) are
    /// both real, and either surface silently losing its half looks exactly
    /// like "the knob does nothing".
    #[test]
    fn parallel_reads_the_flag_and_the_manifest() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let bare_cli = Cli::parse_from(["kglite-mcp-server", "--graph", "g.kgl"]);
        let flag_cli = Cli::parse_from(["kglite-mcp-server", "--graph", "g.kgl", "--parallel"]);

        assert!(
            !boot_parallel(None, &bare_cli).expect("no manifest parses"),
            "neither surface set must leave the engine sequential"
        );
        let bare = manifest_with(tmp.path(), "name: bare\n");
        assert!(
            !boot_parallel(Some(&bare), &bare_cli).expect("bare manifest parses"),
            "an absent key must leave the engine sequential — otherwise the on-arm proves nothing"
        );
        let off = manifest_with(tmp.path(), "name: off\nextensions:\n  parallel: false\n");
        assert!(!boot_parallel(Some(&off), &bare_cli).expect("explicit false parses"));

        assert!(
            boot_parallel(None, &flag_cli).expect("flag alone parses"),
            "--parallel must reach the execution wiring; if this fails the flag is parsed \
             and then dropped"
        );
        let on = manifest_with(tmp.path(), "name: on\nextensions:\n  parallel: true\n");
        assert!(
            boot_parallel(Some(&on), &bare_cli).expect("explicit true parses"),
            "extensions.parallel: true must reach the execution wiring; if this fails the \
             key is parsed and then dropped"
        );
        assert!(
            boot_parallel(Some(&on), &flag_cli).expect("both parse"),
            "flag + manifest is their OR — neither surface can switch the other off"
        );
        assert!(
            boot_parallel(Some(&off), &flag_cli).expect("both parse"),
            "an explicit manifest `false` does not veto the flag; the pair is an OR, and a \
             flag the operator typed on this launch is the more specific statement"
        );
    }

    #[test]
    fn a_malformed_parallel_fails_boot() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let cli = Cli::parse_from(["kglite-mcp-server", "--graph", "g.kgl"]);

        let word = manifest_with(
            tmp.path(),
            "name: word\nextensions:\n  parallel: yes-please\n",
        );
        let error = boot_parallel(Some(&word), &cli)
            .expect_err("a non-boolean value must fail boot, not be ignored")
            .to_string();
        assert!(error.contains("must be a boolean"), "{error}");

        let null = manifest_with(tmp.path(), "name: null\nextensions:\n  parallel:\n");
        assert!(
            boot_parallel(Some(&null), &cli).is_err(),
            "an empty (null) value is a typo, not `false`"
        );

        let listed = manifest_with(
            tmp.path(),
            "name: listed\nextensions:\n  parallel:\n    - true\n",
        );
        assert!(
            boot_parallel(Some(&listed), &cli).is_err(),
            "a list is a typo, not a truthy value"
        );
    }
}
