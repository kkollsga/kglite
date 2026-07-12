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
# docs/history/roadmap-2026H1.md). As of the Piece 4 hard seal, ALL wrapper crates (the wheel +
# the bolt/mcp/c servers) reach the engine through `kglite::api` only, and
# `kglite::graph` is `pub(crate)` — so a below-api reach is also a compile
# error. This grep is the fast, human-readable secondary check: it enforces
# HARD ZERO below-api reaches in every wrapper crate.
#
# `WHEEL_BASELINE` is kept (at 0) as a ratchet so that if the seal is ever
# loosened (e.g. a module re-`pub`'d), the count can only go back to 0:
# any increase fails CI, and dropping below the (already-0) baseline is
# impossible. The history below records how the wheel got from 253 -> 0.

set -euo pipefail
cd "$(dirname "$0")/.."

# A below-api reach is any CROSS-CRATE path into a sealed engine module —
# `graph` or `datasets` (e.g. `kglite::graph::*` / `kglite::datasets::*`, or
# `kglite_core::graph::*` / `kglite_core::datasets::*` from the wheel). The
# curated `kglite_core::api::datasets::*` path does NOT match (the crate prefix
# is followed by `api`, not `datasets`), so it is correctly allowed.
# Since the hard seal (Piece 4), `kglite::graph` is `pub(crate)`, so any such
# reach is ALSO a compile error — this grep is the fast, human-readable
# secondary check. (`crate::graph::*` is NOT counted: that's a wrapper's own
# graph-namespaced modules — e.g. the wheel's pyapi/embedder/languages.)
# Strip `//` line + `/* */` block comments first so a path named in prose
# never counts. grep exits 1 on no match; `|| true` tolerates that.
count_reaches() {
	local dir="$1"
	find "$dir" -name '*.rs' -exec cat {} + 2>/dev/null \
		| perl -0pe 's{/\*.*?\*/}{}gs; s{//[^\n]*}{}g' \
		| { grep -cE "(kglite|kglite_core)::(graph|datasets)::" || true; }
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
# explore, dir_graph, handle, io::file, blueprint — onto api paths)
# -> 0 (Piece 4 HARD SEAL: storage/durability/embedding lifted to
#       api::storage/api::durable/api::io; cypher pipeline + Selection /
#       TemporalContext lifted; glob deleted; `kglite::graph` demoted to
#       pub(crate) — every wrapper now reaches the engine ONLY through
#       kglite::api, compiler-enforced. This gate is now belt-and-suspenders.)
# (superseded baseline note) -> 27 (long-tail batch 2: lift io export/ntriples, spatial/temporal,
# introspection-compute, schema type family, validate_graph). The
# remaining 27 are the storage-backend + durability + embedding cluster
# (GraphBackend/DiskGraph/MappedGraph/EmbeddingStore/recording/wal/
# subgraph_streaming) — they need a high-level api design (Piece 4), not
# a raw lift of the storage/WAL internals.
WHEEL_BASELINE=0

fail=0

# --- 1. Server crates: hard zero -------------------------------------------
for crate in kglite-bolt-server kglite-mcp-server kglite-c; do
	n=$(count_reaches "crates/$crate/src")
	if [ "$n" -ne 0 ]; then
		echo "FAIL: crates/$crate reaches below kglite::api ($n times) — must be 0."
		echo "      Offending lines:"
		{ grep -rnE "(kglite|kglite_core)::(graph|datasets)::" "crates/$crate/src" \
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
	echo "      (see docs/history/roadmap-2026H1.md / CLAUDE.md boundary principle)."
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
