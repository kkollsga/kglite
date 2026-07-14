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
tar -xzf "$archive" -C "$tmp"
cargo check \
  --manifest-path "$tmp/kglite-${version}/Cargo.toml" \
  --features parallel-bz2 \
  --lib
