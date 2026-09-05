#!/usr/bin/env python3
"""Bounded native CREATE attribution; only run while owning the measurement lease.

Release standalone codegen uses the standing shared target, temporary package and
current source. Output is compressed under the existing bounded benchmark tier.
No extension installation, historical source build, or root manifest mutation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import lzma
import pathlib
import subprocess
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[3]
SOURCE = pathlib.Path(__file__).with_suffix(".rs")


def command(*args: str, capture: bool = False) -> str:
    result = subprocess.run(
        args, cwd=ROOT, check=False, text=True, stdout=subprocess.PIPE if capture else None, timeout=900
    )
    result.check_returncode()
    return result.stdout if capture else ""


def sha(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes", type=int, nargs="+", default=[1000, 10000, 30000])
    parser.add_argument("--widths", type=int, nargs="+", default=[1])
    parser.add_argument("--events", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--reverse", action="store_true")
    parser.add_argument("--discriminator", action="store_true")
    parser.add_argument("--out", type=pathlib.Path, required=True)
    args = parser.parse_args()
    if not (
        1 <= args.events <= 200
        and 0 <= args.warmup <= 20
        and len(args.nodes) <= 3
        and len(args.widths) <= 3
        and all(1 <= n <= 30000 for n in args.nodes)
        and all(n in [1, 4, 16] for n in args.widths)
    ):
        parser.error("invalid or unbounded parameters")
    if not args.discriminator and args.widths != [1]:
        parser.error("widths require the approved discriminator stage")
    out = args.out.resolve()
    if out.exists() or out.suffixes[-2:] != [".json", ".xz"] or not out.is_relative_to(ROOT / "dev-docs/bench/out"):
        parser.error("output must be a new .json.xz under dev-docs/bench/out")
    target = ROOT / "target"
    if not target.is_symlink() or not target.resolve().is_dir():
        parser.error("restore the standing target symlink")
    command("make", "check-free-space")
    with tempfile.TemporaryDirectory(prefix="session-residual-", dir=ROOT / "dev-docs/temp") as name:
        package = pathlib.Path(name)
        (package / "target").symlink_to(target.resolve(), target_is_directory=True)
        manifest = f"""[package]
name = "kglite-session-residual"
version = "0.0.0"
edition = "2021"
[workspace]
[dependencies]
kglite = {{ path = {json.dumps(str(ROOT / "crates/kglite"))}, default-features = false }}
serde_json = "1.0.150"
[[bin]]
name = "session_create_residual"
path = {json.dumps(str(SOURCE))}
[profile.release]
lto = "thin"
codegen-units = 1
strip = "symbols"
"""
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
            r["executable"]
            for r in artifacts
            if r.get("reason") == "compiler-artifact"
            and r.get("target", {}).get("name") == "session_create_residual"
            and r.get("executable")
        ]
        assert len(binaries) == 1
        binary = pathlib.Path(binaries[0])
        assert binary.resolve().is_relative_to(target.resolve())
        captures = []
        for nodes in args.nodes:
            for width in args.widths:
                result = json.loads(
                    command(
                        str(binary),
                        str(nodes),
                        str(width),
                        str(args.events),
                        str(args.warmup),
                        str(int(args.reverse)),
                        "width" if args.discriminator else "base",
                        capture=True,
                    )
                )
                assert len(result["records"]) == 2 and all(r["oracle"]["passed"] for r in result["records"])
                captures.append(result)
        result = {
            "head": command("git", "rev-parse", "HEAD", capture=True).strip(),
            "source_sha256": sha(SOURCE),
            "wrapper_sha256": sha(pathlib.Path(__file__)),
            "binary_sha256": sha(binary),
            "manifest": manifest,
            "resolved_lock": (package / "Cargo.lock").read_text(),
            "rustc": command("rustc", "--version", capture=True).strip(),
            "status": command("git", "status", "--porcelain", capture=True).splitlines(),
            "diff_sha256": hashlib.sha256(command("git", "diff", "HEAD", capture=True).encode()).hexdigest(),
            "discriminator": args.discriminator,
            "captures": captures,
        }
        raw = json.dumps(result, indent=2).encode()
        encoded = lzma.compress(raw)
        assert lzma.decompress(encoded) == raw
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("xb") as f:
            f.write(encoded)
        print(
            json.dumps(
                {
                    "out": str(out),
                    "raw_sha256": hashlib.sha256(raw).hexdigest(),
                    "archive_sha256": sha(out),
                    "raw_bytes": len(raw),
                    "archive_bytes": len(encoded),
                }
            )
        )


if __name__ == "__main__":
    main()
