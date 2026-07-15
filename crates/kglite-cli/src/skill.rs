//! Offline installer for the canonical code-review Agent Skill.

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use clap::{Subcommand, ValueEnum};

const SKILL_DIR: &str = "kglite-code-review";
const MARKER: &str = ".kglite-managed";
const FILES: &[(&str, &str)] = &[
    (
        "SKILL.md",
        include_str!("../skills/kglite-code-review/SKILL.md"),
    ),
    (
        "references/queries.md",
        include_str!("../skills/kglite-code-review/references/queries.md"),
    ),
    (
        "references/public-repositories.md",
        include_str!("../skills/kglite-code-review/references/public-repositories.md"),
    ),
    (
        "references/mcp-upgrade.md",
        include_str!("../skills/kglite-code-review/references/mcp-upgrade.md"),
    ),
];

#[derive(Subcommand, Debug)]
pub enum SkillCommand {
    /// Install the bundled skill for Codex, Claude Code, or both.
    Install {
        /// Host to install for. Repeat for several; omitted installs both.
        #[arg(long, value_enum)]
        host: Vec<Host>,
        /// Install under the current project instead of the user home.
        #[arg(long)]
        project: bool,
        /// Print planned changes without writing files.
        #[arg(long)]
        dry_run: bool,
    },
    /// Remove a skill previously installed by this command.
    Uninstall {
        /// Host to uninstall from. Repeat for several; omitted checks both.
        #[arg(long, value_enum)]
        host: Vec<Host>,
        /// Remove from the current project instead of the user home.
        #[arg(long)]
        project: bool,
        /// Print planned changes without writing files.
        #[arg(long)]
        dry_run: bool,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
pub enum Host {
    Codex,
    Claude,
}

impl Host {
    fn label(self) -> &'static str {
        match self {
            Self::Codex => "codex",
            Self::Claude => "claude",
        }
    }
}

pub fn run(command: &SkillCommand) -> Result<()> {
    let cwd = std::env::current_dir()?;
    let home = home_dir()?;
    match command {
        SkillCommand::Install {
            host,
            project,
            dry_run,
        } => install(hosts(host), *project, *dry_run, &home, &cwd),
        SkillCommand::Uninstall {
            host,
            project,
            dry_run,
        } => uninstall(hosts(host), *project, *dry_run, &home, &cwd),
    }
}

fn hosts(selected: &[Host]) -> Vec<Host> {
    if selected.is_empty() {
        vec![Host::Codex, Host::Claude]
    } else {
        selected.to_vec()
    }
}

fn home_dir() -> Result<PathBuf> {
    std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map(PathBuf::from)
        .ok_or_else(|| anyhow::anyhow!("could not determine user home directory"))
}

fn destination(host: Host, project: bool, home: &Path, cwd: &Path) -> PathBuf {
    let root = if project { cwd } else { home };
    match host {
        Host::Codex => root.join(".codex/skills").join(SKILL_DIR),
        Host::Claude => root.join(".claude/skills").join(SKILL_DIR),
    }
}

fn install(hosts: Vec<Host>, project: bool, dry_run: bool, home: &Path, cwd: &Path) -> Result<()> {
    for host in hosts {
        let dest = destination(host, project, home, cwd);
        println!(
            "{}install {} skill at {}",
            if dry_run { "would " } else { "" },
            host.label(),
            dest.display()
        );
        if dry_run {
            continue;
        }
        if dest.exists() && !dest.join(MARKER).is_file() {
            anyhow::bail!(
                "refusing to replace unmanaged skill directory: {}",
                dest.display()
            );
        }
        write_skill_atomically(&dest)?;
    }
    Ok(())
}

fn uninstall(
    hosts: Vec<Host>,
    project: bool,
    dry_run: bool,
    home: &Path,
    cwd: &Path,
) -> Result<()> {
    for host in hosts {
        let dest = destination(host, project, home, cwd);
        if !dest.exists() {
            println!(
                "{} skill is not installed at {}",
                host.label(),
                dest.display()
            );
            continue;
        }
        if !dest.join(MARKER).is_file() {
            anyhow::bail!(
                "refusing to remove unmanaged skill directory: {}",
                dest.display()
            );
        }
        println!(
            "{}remove {} skill at {}",
            if dry_run { "would " } else { "" },
            host.label(),
            dest.display()
        );
        if !dry_run {
            fs::remove_dir_all(&dest)
                .with_context(|| format!("could not remove {}", dest.display()))?;
        }
    }
    Ok(())
}

fn write_skill_atomically(dest: &Path) -> Result<()> {
    let parent = dest
        .parent()
        .ok_or_else(|| anyhow::anyhow!("skill destination has no parent"))?;
    fs::create_dir_all(parent)?;
    let temp = parent.join(format!(".{SKILL_DIR}.tmp-{}", std::process::id()));
    if temp.exists() {
        fs::remove_dir_all(&temp)?;
    }
    for (relative, body) in FILES {
        let path = temp.join(relative);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(&path, body)?;
    }
    fs::write(temp.join(MARKER), env!("CARGO_PKG_VERSION"))?;
    if dest.exists() {
        let backup = parent.join(format!(".{SKILL_DIR}.backup-{}", std::process::id()));
        if backup.exists() {
            fs::remove_dir_all(&backup)?;
        }
        fs::rename(dest, &backup)?;
        if let Err(error) = fs::rename(&temp, dest) {
            let _ = fs::rename(&backup, dest);
            return Err(error)
                .with_context(|| format!("could not activate skill at {}", dest.display()));
        }
        fs::remove_dir_all(backup)?;
        return Ok(());
    }
    fs::rename(&temp, dest)
        .with_context(|| format!("could not activate skill at {}", dest.display()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn install_is_idempotent_and_uninstall_is_clean_at_both_scopes() {
        let tmp = tempfile::tempdir().unwrap();
        let home = tmp.path().join("home");
        let project = tmp.path().join("project");
        fs::create_dir_all(&home).unwrap();
        fs::create_dir_all(&project).unwrap();

        for project_scope in [false, true] {
            install(
                vec![Host::Codex, Host::Claude],
                project_scope,
                false,
                &home,
                &project,
            )
            .unwrap();
            // Re-install replaces the managed artifact rather than duplicating it.
            install(
                vec![Host::Codex, Host::Claude],
                project_scope,
                false,
                &home,
                &project,
            )
            .unwrap();
            for host in [Host::Codex, Host::Claude] {
                let dest = destination(host, project_scope, &home, &project);
                let body = fs::read_to_string(dest.join("SKILL.md")).unwrap();
                assert!(body.starts_with("---\nname: kglite-code-review\n"));
                assert!(dest.join(MARKER).is_file());
                assert!(dest.join("references/queries.md").is_file());
            }
            uninstall(
                vec![Host::Codex, Host::Claude],
                project_scope,
                false,
                &home,
                &project,
            )
            .unwrap();
            assert!(!destination(Host::Codex, project_scope, &home, &project).exists());
            assert!(!destination(Host::Claude, project_scope, &home, &project).exists());
        }
    }

    #[test]
    fn dry_run_does_not_write_and_unmanaged_directory_is_protected() {
        let tmp = tempfile::tempdir().unwrap();
        install(vec![Host::Codex], false, true, tmp.path(), tmp.path()).unwrap();
        let dest = destination(Host::Codex, false, tmp.path(), tmp.path());
        assert!(!dest.exists());

        fs::create_dir_all(&dest).unwrap();
        fs::write(dest.join("SKILL.md"), "user-owned").unwrap();
        let error = install(vec![Host::Codex], false, false, tmp.path(), tmp.path()).unwrap_err();
        assert!(error.to_string().contains("unmanaged skill directory"));
    }

    #[test]
    fn embedded_artifact_matches_checked_in_files() {
        for (relative, embedded) in FILES {
            let checked_in = Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../skills/kglite-code-review")
                .join(relative);
            assert_eq!(fs::read_to_string(checked_in).unwrap(), *embedded);
        }
    }
}
