#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

version="$(cargo metadata --no-deps --format-version 1 | python3 -c \
  'import json, sys; print(next(p["version"] for p in json.load(sys.stdin)["packages"] if p["name"] == "kglite"))')"
archive="target/package/kglite-${version}.crate"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

cargo package -p kglite --locked --allow-dirty --no-verify
mkdir -p "$tmp/crates" "$tmp/tests/fixtures"
tar -xzf "$archive" -C "$tmp"
mv "$tmp/kglite-${version}" "$tmp/crates/kglite"
cargo check \
  --manifest-path "$tmp/crates/kglite/Cargo.toml" \
  --features parallel-bz2 \
  --lib

# Compile and run the packaged crate as a real external dependency. Keeping
# the fixture's relative path layout intact means this checks the local source
# tree and the extracted crates.io archive with the same Cargo.toml.
cp -R tests/fixtures/rust-embed-consumer "$tmp/tests/fixtures/"
cargo run \
  --manifest-path "$tmp/tests/fixtures/rust-embed-consumer/Cargo.toml" \
  --locked \
  --quiet
