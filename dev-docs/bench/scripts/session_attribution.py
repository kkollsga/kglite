#!/usr/bin/env python3
"""Phase 5 native attribution: build only on the coordinator's measurement lease.

No root manifest edits or example replacement. A standalone temporary Cargo
package points its binary at the reviewed Rust source and depends on this tree's
engine only. Its target symlink uses the standing shared cache. Temporary package
and lock are removed on exit; captures belong to dev-docs/bench/out cleanup.

Small preflight:
uv run --no-sync python dev-docs/bench/scripts/session_attribution.py \
    --nodes 64 --edges 192 --counts 32 --window 16 --fixed-nodes 8 --repeats 1 \
    --out dev-docs/bench/out/session-attribution-preflight.json

Default full probe covers both CREATE and fixed SET, 10k initial nodes/30k edges,
1k/10k/30k commits, two fresh repeats each. It does not exercise WAL or Bolt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import subprocess
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[3]
SOURCE = pathlib.Path(__file__).with_suffix(".rs")


def command(*args: str, capture: bool = False, timeout: int | None = None) -> str:
    result = subprocess.run(
        args, cwd=ROOT, check=False, text=True, stdout=subprocess.PIPE if capture else None, timeout=timeout
    )
    if result.returncode:
        if capture:
            print(result.stdout, flush=True)
        result.check_returncode()
    return result.stdout if capture else ""


def digest(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes", type=int, default=10_000)
    parser.add_argument("--edges", type=int, default=30_000)
    parser.add_argument("--counts", type=int, nargs="+", default=[1000, 10_000, 30_000])
    parser.add_argument("--window", type=int, default=1000)
    parser.add_argument("--fixed-nodes", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--timeout-s", type=int, default=600, help="hang ceiling per commit-count subprocess")
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--control", choices=["owned", "bulk", "held", "drop-half", "fresh-holder", "normal"])
    parser.add_argument("--bulk", type=int, default=0)
    args = parser.parse_args()
    source = SOURCE.with_name("session_controls.rs") if args.control else SOURCE
    if args.bulk < 0 or args.bulk > 100_000 or (args.bulk and args.control != "bulk"):
        parser.error("bulk prefill requires bulk control and a bounded count")
    out = args.out.resolve()
    roots = [(ROOT / "dev-docs/bench/out").resolve(), (ROOT / "dev-docs/temp").resolve()]
    if not any(out.is_relative_to(root) for root in roots) or out.exists():
        parser.error("--out must be a new file under dev-docs/bench/out or dev-docs/temp")
    if len(args.counts) > 3 or len(set(args.counts)) != len(args.counts):
        parser.error("choose one to three distinct commit counts")
    if not (
        0 < args.timeout_s <= 3600
        and 0 < args.nodes <= 100_000
        and 0 <= args.edges <= 300_000
        and 0 < args.fixed_nodes <= args.nodes
        and 0 < args.repeats <= 10
        and args.window > 0
        and all(
            0 < n <= 100_000 and args.window <= n and (n + args.window - 1) // args.window <= 1000 for n in args.counts
        )
    ):
        parser.error("invalid or unbounded driver parameters")
    target = ROOT / "target"
    if not target.is_symlink() or not target.resolve().is_dir():
        parser.error("restore the standing target symlink before building")
    command("make", "check-free-space")
    scratch = ROOT / "dev-docs/temp"
    scratch.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="session-attribution-", dir=scratch) as name:
        package = pathlib.Path(name)
        (package / "target").symlink_to(target.resolve(), target_is_directory=True)
        manifest = "\n".join(
            [
                "[package]",
                'name = "kglite-session-attribution"',
                'version = "0.0.0"',
                'edition = "2021"',
                "[workspace]",
                "[dependencies]",
                f"kglite = {{ path = {json.dumps(str(ROOT / 'crates/kglite'))}, default-features = false }}",
                'serde_json = "1.0.150"',
                "[[bin]]",
                'name = "session_attribution"',
                f"path = {json.dumps(str(source))}",
                "[profile.release]",
                'lto = "thin"',
                "codegen-units = 1",
                'strip = "symbols"',
                "",
            ]
        )
        (package / "Cargo.toml").write_text(manifest)
        build = command(
            "cargo",
            "build",
            "--release",
            "--offline",
            "--message-format=json",
            "--manifest-path",
            str(package / "Cargo.toml"),
            capture=True,
        )
        artifacts = [json.loads(line) for line in build.splitlines() if line.strip()]
        binaries = [
            item["executable"]
            for item in artifacts
            if item.get("reason") == "compiler-artifact"
            and item.get("target", {}).get("name") == "session_attribution"
            and item.get("executable")
        ]
        if len(binaries) != 1:
            raise RuntimeError("build did not identify exactly one measurement executable")
        binary = pathlib.Path(binaries[0])
        if not binary.is_file() or not binary.resolve().is_relative_to(target.resolve()):
            raise RuntimeError("build artifact is missing or outside the standing shared target")
        records = []
        for count in args.counts:
            raw = command(
                str(binary),
                str(args.nodes),
                str(args.edges),
                str(count),
                str(args.window),
                str(args.fixed_nodes),
                str(args.repeats),
                *([args.control, str(args.bulk), str(package / "durable")] if args.control else []),
                capture=True,
                timeout=args.timeout_s,
            )
            record = json.loads(raw)
            if not record["materialization_sanity"]["passed"]:
                raise RuntimeError("materialization sanity failed")
            if len(record["runs"]) != args.repeats * (1 if args.control else 2) or not all(
                r["oracle"]["passed"] for r in record["runs"]
            ):
                raise RuntimeError("driver returned incomplete runs or failed oracles")
            records.append(record)
        result = {
            "driver_sha256": digest(source),
            "shared_driver_sha256": digest(SOURCE),
            "control": args.control,
            "wrapper_sha256": digest(pathlib.Path(__file__)),
            "binary_sha256": digest(binary),
            "manifest": manifest,
            "resolved_lock_sha256": digest(package / "Cargo.lock"),
            "resolved_lock": (package / "Cargo.lock").read_text(),
            "rustc": command("rustc", "--version", capture=True).strip(),
            "workspace_status": command("git", "status", "--porcelain", capture=True).splitlines(),
            "git_head": command("git", "rev-parse", "HEAD", capture=True).strip(),
            "tracked_diff_sha256": hashlib.sha256(command("git", "diff", "HEAD", capture=True).encode()).hexdigest(),
            "captures": records,
        }
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("x") as handle:
            json.dump(result, handle, indent=2)
            handle.write("\n")
    print(f"Saved verified driver results to {out}")


if __name__ == "__main__":
    main()
