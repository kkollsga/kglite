"""Bounded standalone release attribution; no root manifests or API changes.

Uses the current production module by path, with the standing shared target.
Corpora and Cargo package/lock are temporary and deleted at exit. Captures must
be written below dev-docs/bench/out or results. End-to-end engine measurements
remain the acceptance gate; this standalone codegen scope only selects a
candidate strategy.
"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SOURCE = Path(__file__).with_suffix(".rs")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes", type=int, choices=[5000, 20_000, 100_000], required=True)
    parser.add_argument("--deltas", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    assert args.out.resolve().is_relative_to(ROOT / "dev-docs/bench") and not args.out.exists()
    deltas = [int(value) for value in args.deltas.split(",")]
    assert len(deltas) <= 9 and all(0 < value <= args.nodes for value in deltas)
    target = ROOT / "target"
    assert target.is_symlink() and target.resolve().is_dir()
    subprocess.run(["make", "check-free-space"], cwd=ROOT, check=True)
    with tempfile.TemporaryDirectory(prefix="kglite-text-refresh-") as temp:
        package = Path(temp)
        (package / "target").symlink_to(target.resolve(), target_is_directory=True)
        manifest = f"""[package]
name = "kglite-text-refresh-attribution"
version = "0.0.0"
edition = "2021"
[workspace]
[dependencies]
rustc-hash = "2"
serde = {{ version = "1", features = ["derive", "rc"] }}
serde_json = "1.0.150"
[[bin]]
name = "text_refresh_attribution"
path = {json.dumps(str(SOURCE))}
[profile.release]
lto = "thin"
codegen-units = 1
strip = "symbols"
"""
        (package / "Cargo.toml").write_text(manifest)
        result = subprocess.run(
            [
                "cargo",
                "build",
                "--release",
                "--offline",
                "--message-format=json",
                "--manifest-path",
                str(package / "Cargo.toml"),
            ],
            cwd=ROOT,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
        )
        artifacts = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
        executable = next(item["executable"] for item in artifacts if item.get("executable"))
        rng = np.random.default_rng(20_260_825)
        lengths = rng.integers(100, 301, size=args.nodes)
        vocab = np.array([f"w{i:05d}" for i in range(65_536)], dtype=object)
        probabilities = 1.0 / np.arange(1, 65_537, dtype=np.float64)
        probabilities /= probabilities.sum()
        tokens = vocab[rng.choice(len(vocab), size=int(lengths.sum()), p=probabilities)]
        bodies, offset = [], 0
        for length in lengths:
            bodies.append(" ".join(tokens[offset : offset + length]))
            offset += length
        corpus = package / "corpus.json"
        corpus.write_text(json.dumps(bodies))
        del bodies, tokens
        result = subprocess.run(
            [executable, str(corpus), args.deltas], cwd=ROOT, check=True, text=True, stdout=subprocess.PIPE
        )
        paths = [
            SOURCE,
            Path(__file__),
            ROOT / "crates/kglite/src/graph/algorithms/text_index/mod.rs",
            ROOT / "crates/kglite/src/graph/algorithms/text_index/batch.rs",
        ]
        args.out.write_text(
            json.dumps(
                {
                    "records": json.loads(result.stdout),
                    "sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
                    "profile": "standalone release, current production text core by path",
                    "primitive_build_exit": 0,
                    "primitive_measure_exit": 0,
                },
                indent=2,
            )
            + "\n"
        )


if __name__ == "__main__":
    main()
