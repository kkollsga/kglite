#!/usr/bin/env bash
#
# check_api_chokepoint.sh — enforce the kglite::api single-chokepoint boundary.
#
# Two-tier binding architecture (CLAUDE.md → "Two-tier standardization
# architecture"): every downstream wrapper reaches the engine through the
# curated, semver-stable `kglite::api::*` surface — never the raw
# `kglite_core::graph::*` (or `kglite::graph::*`) module tree.
#
# This script is the regression RATCHET for the api-sealing effort (see
# roadmap.md). It does two things:
#
#   1. The Rust-side server crates (bolt / mcp / c) are already clean — they
#      go through `kglite::api` exclusively. Enforce HARD ZERO below-api
#      reaches there, so they never regress.
#
#   2. The Python wheel (kglite-py) historically reaches deep below api (its
#      fluent API is implemented across the crate boundary). We can't fix
#      that in one shot, so we FREEZE the current reach count as a baseline
#      and fail on any INCREASE. The number can only shrink — as roadmap
#      Pieces 2-4 land, lower WHEEL_BASELINE to match.
#
# When the count drops below the baseline, the script tells you to lower the
# baseline (so the ratchet keeps biting). When it exceeds the baseline, CI
# fails: a new below-api reach crept in — route it through kglite::api
# instead, or lift the needed item into api first.

set -euo pipefail
cd "$(dirname "$0")/.."

# Engine submodules that live BELOW api. Excludes the wheel-local modules
# that legitimately sit under `crate::graph::` in kglite-py (pyapi, embedder,
# languages) — those are the wheel's own PyO3 code, not engine reaches.
ENGINE='(core|algorithms|features|mutation|storage|schema|session|io|introspection|explore|dir_graph|handle|wal|blueprint)'

# A reach = `crate::graph::<engine>` or `<crate>::graph::<engine>`, where the
# crate alias is `kglite` (bolt/mcp/c) or `kglite_core` (the wheel). Comment
# and doc-comment lines (`//` / `///`) are excluded — they reference paths in
# prose, not code.
# Count code reaches. Strip `//` line comments AND `/* */` block comments
# before matching, so a path mentioned in prose/doc-comments never counts
# (a bare `grep -v '//'` missed block comments and trailing comments). grep
# exits 1 on no match; `|| true` tolerates that (a clean crate has zero).
count_reaches() {
	local dir="$1"
	find "$dir" -name '*.rs' -exec cat {} + 2>/dev/null \
		| perl -0pe 's{/\*.*?\*/}{}gs; s{//[^\n]*}{}g' \
		| { grep -cE "(crate|kglite|kglite_core)::graph::$ENGINE" || true; }
}

# The wheel's frozen baseline. Lower this as roadmap Pieces 2-4 migrate
# reaches onto kglite::api. NEVER raise it.
# History: 253 (Piece 1) -> 153 (Piece 2: algorithms, bulk mutation +
# reports, timeseries, GraphRead, InternedKey lifted) -> 137 (Piece 3a/3b:
# Selection data model + Selection-coupled capabilities — vector_search,
# create_connections, set_ops, subgraph, infer_selection_node_type)
# -> 85 (Piece 3c: the shared selection-based query-primitive layer
# (filtering/traversal/calculations/statistics/data_retrieval/
# pattern_matching/value_operations) exposed via api::fluent)
# -> 66 (long-tail batch 1: migrate already-in-api clusters — session,
# explore, dir_graph, handle, io::file, blueprint — onto api paths).
WHEEL_BASELINE=66

fail=0

# --- 1. Server crates: hard zero -------------------------------------------
for crate in kglite-bolt-server kglite-mcp-server kglite-c; do
	n=$(count_reaches "crates/$crate/src")
	if [ "$n" -ne 0 ]; then
		echo "FAIL: crates/$crate reaches below kglite::api ($n times) — must be 0."
		echo "      Offending lines:"
		{ grep -rnE "(crate|kglite|kglite_core)::graph::$ENGINE" "crates/$crate/src" \
			| grep -vE ':[[:space:]]*//' || true; } | sed 's/^/        /'
		fail=1
	else
		echo "ok:   crates/$crate — 0 below-api reaches"
	fi
done

# --- 2. Wheel: frozen ratchet ----------------------------------------------
wheel=$(count_reaches "crates/kglite-py/src")
if [ "$wheel" -gt "$WHEEL_BASELINE" ]; then
	echo "FAIL: crates/kglite-py below-api reaches grew to $wheel (baseline $WHEEL_BASELINE)."
	echo "      A new kglite_core::graph:: reach crept in. Route it through"
	echo "      kglite::api instead, or lift the needed item into api first"
	echo "      (see roadmap.md / CLAUDE.md boundary principle)."
	fail=1
elif [ "$wheel" -lt "$WHEEL_BASELINE" ]; then
	echo "FAIL: crates/kglite-py below-api reaches dropped to $wheel (baseline $WHEEL_BASELINE)."
	echo "      A lift reduced the count — lower WHEEL_BASELINE to $wheel in"
	echo "      scripts/check_api_chokepoint.sh (in the same change) so the"
	echo "      ratchet stays tight. The baseline must track the floor exactly,"
	echo "      otherwise the freed slack lets new below-api reaches creep back."
	fail=1
else
	echo "ok:   crates/kglite-py — $wheel below-api reaches (at baseline)."
fi

if [ "$fail" -ne 0 ]; then
	exit 1
fi
echo "api chokepoint: OK"
