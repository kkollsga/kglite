"""Phase 5 crunch-point parity tests.

Guards the columnar-cleanup + per-backend-impls phase of the 0.8.0
storage refactor. Tests here:

- **graph.copy() CoW correctness** — mutating a copy leaves the
  original unchanged on the memory and mapped backends. This is the Phase 0
  crunch-point re-asserted after Phase 5's per-backend impls + ColumnStore
  split. Disk copy ownership has a separate storage-lifecycle follow-up.
- **binary-size regression gate** — the release extension stays under
  the +10% budget relative to the current per-platform baseline.

Marker assignment is per-function so the expensive checks stay opt-in:

  - `test_graph_copy_cow_correctness_*` — `@pytest.mark.parity`
    (functional, needs backend setup).
  - `test_binary_size_regression` — `@pytest.mark.binary_size`
    (needs the release extension built; CI's `python-tests` job
    already builds a release wheel with maturin, so it plugs in there).

Dead code is the workspace clippy CI job's gate (`-D warnings`), not a test
here.

Run: pytest tests/test_phase5_parity.py -m parity        (functional)
     pytest tests/test_phase5_parity.py -m binary_size   (release-build gate)
"""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import pytest

from kglite import KnowledgeGraph

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.parity
def test_graph_copy_cow_correctness_memory():
    """Mutating the copy does not affect the original (in-memory backend)."""

    kg = KnowledgeGraph()
    df = pd.DataFrame([{"pid": 1, "name": "Alice", "age": 30}, {"pid": 2, "name": "Bob", "age": 25}])
    kg.add_nodes(df, "Person", "pid", "name")

    kg2 = kg.copy()
    kg2.add_nodes(
        pd.DataFrame([{"pid": 1, "name": "Alice Updated", "age": 99}]),
        "Person",
        "pid",
        "name",
        conflict_handling="update",
    )

    orig = kg.cypher("MATCH (n:Person) WHERE n.id = 1 RETURN n.age AS age")
    mod = kg2.cypher("MATCH (n:Person) WHERE n.id = 1 RETURN n.age AS age")

    orig_rows = [dict(r) for r in orig]
    mod_rows = [dict(r) for r in mod]

    assert orig_rows == [{"age": 30}], f"original mutated unexpectedly: {orig_rows}"
    assert mod_rows == [{"age": 99}], f"copy update did not apply: {mod_rows}"


@pytest.mark.parity
def test_graph_copy_cow_correctness_mapped():
    """Mutating the copy does not affect the original (mapped backend)."""

    kg = KnowledgeGraph(storage="mapped")
    df = pd.DataFrame([{"pid": 1, "name": "Alice", "age": 30}, {"pid": 2, "name": "Bob", "age": 25}])
    kg.add_nodes(df, "Person", "pid", "name")

    kg2 = kg.copy()
    kg2.add_nodes(
        pd.DataFrame([{"pid": 1, "name": "Alice Updated", "age": 99}]),
        "Person",
        "pid",
        "name",
        conflict_handling="update",
    )

    orig = [dict(r) for r in kg.cypher("MATCH (n:Person) WHERE n.id = 1 RETURN n.age AS age")]
    mod = [dict(r) for r in kg2.cypher("MATCH (n:Person) WHERE n.id = 1 RETURN n.age AS age")]

    assert orig == [{"age": 30}], f"mapped original mutated: {orig}"
    assert mod == [{"age": 99}], f"mapped copy update lost: {mod}"


#: Per-platform release library size baseline, as (size in bytes, version it
#: describes). The Linux ELF is larger than the macOS Mach-O for the same
#: source (linker, debug-info and strip defaults differ); CI gates Linux, local
#: development gates macOS. `make refresh-release-constants` rewrites both rows
#: on a macOS host: macOS from the host release build at the version being cut,
#: Linux from the `kglite/kglite.abi3.so` member of the newest manylinux2014
#: x86_64 wheel already on PyPI, whose file name and SHA-256 follow the table.
BINARY_SIZE_BASELINES = {
    "darwin": (32_160_448, "0.21.4"),  # host release build
    "linux": (39_802_488, "0.21.3"),  # published manylinux2014 x86_64 wheel member
}
LINUX_SIZE_WHEEL = "kglite-0.21.3-cp312-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl"
LINUX_SIZE_WHEEL_SHA256 = "c498497fda81660a69488c990806115b34ca096c4e2e8590e8ef9f39f4e5c643"


@pytest.mark.binary_size
def test_binary_size_regression():
    """Release library size stays under a +10% budget over the per-platform
    baseline.

    Baseline history:
      - Phase 4 exit:  6,996,688 bytes (≈6.67 MB, macOS).
      - 0.9.0:        23,535,664 bytes (≈22.4 MB, macOS). Multi-mode
                      storage, spatial, timeseries, code-tree, MCP,
                      Cypher dialect coverage all landed in the 0.8.x sweep.
      - 0.9.52:       35,925,104 bytes (≈34.3 MB, macOS .dylib).
                      59,529,016 bytes (≈56.8 MB, Linux .so) —
                      added when the first CI run on 0.9.52 surfaced
                      the platform divergence. Growth between 0.9.0
                      and 0.9.52 (~52% on macOS) is concentrated in:
                        * 14 tree-sitter grammars (Dart added 0.9.51,
                          Swift 0.9.40, PHP/HTML/CSS in the 0.9.2x
                          range — each grammar is ~0.5-1 MB);
                        * fastembed feature default-on for the
                          kglite-mcp-server binary build (ort runtime
                          + hf-hub native TLS path, ~3-4 MB);
                        * mcp-methods 0.3.x server-feature evolution;
                        * dataset-loader workspace crates with
                          their own dependency closures.
      - 0.10.26:      39,319,984 bytes (≈37.5 MB, macOS .dylib).
                      The kglite-mcp-server *library* is now bundled
                      into the wheel (its `run` statically linked into
                      the extension, so `pip install kglite` ships the
                      MCP server). It shares the one kglite engine — no
                      duplication — but pulls the server's own closure
                      (mcp-methods, rmcp, hyper/hyper-util, clap,
                      tracing-subscriber) into the cdylib: ~3 MB net on
                      macOS after strip, more on Linux (no strip).
      - 0.13.0:       41,334,128 bytes (≈39.4 MB, macOS .dylib).
                      Checked persistence decoders, disk generation and
                      snapshot ownership, writer leases, complete C-ABI panic
                      boundaries, bounded executor guards, and guarded lazy
                      result materialization account for the growth since the
                      prior published baseline.
      - post-0.13.4:  19,767,648 bytes (≈18.9 MB, macOS .dylib) — a 53%
                      SHRINK: the in-tree code_tree builder and all 15
                      tree-sitter grammars moved to the standalone codingest
                      project. Baseline deliberately recaptured downward so
                      the +10% budget guards the new, smaller binary.
      - post-0.13.4b: 18,857,776 bytes (≈18.0 MB, macOS .dylib) — the
                      sec/wikidata dataset loaders moved to the
                      kglite-datasets project (zip/quick-xml gone; the
                      remaining ureq/rustls belong to the bundled MCP
                      server via mcp-methods, not the engine).

      - 0.13.1:       41,367,232 bytes (≈39.5 MB). The 33 KB increase adds
                      fused grouped/global count operators, mutation-safe
                      in-memory peer histograms, and fixed-path trail pruning.

      - 0.13.2:       41,400,304 bytes (≈39.5 MB). Added node and relationship
                      text-predicate matchers plus index-aware optimizer routing.


      - 0.13.3:       41,466,176 bytes (≈39.5 MB). Added independent graph-copy
                      identity and disk writer-lineage transfer; refreshed native
                      dependencies. Net growth: 65,872 bytes (0.16%).


      - 0.13.4:       42,310,576 bytes (≈40.4 MB). Bundled the shared Rust
                      CLI (including code-review skill and code-tree commands)
                      into the Python extension, and added Postcard alongside
                      the retained legacy bincode reader. Net growth: 844,400
                      bytes (2.04%).


      - 0.14.2:       18,907,376 bytes (≈18.0 MB). Inline-record endpoint
                      validation, write-provenance propagation, and nested
                      value preservation added 49,600 bytes (0.26%).


      - 0.14.3:       18,940,768 bytes (≈18.1 MB). Postcard-only
                      persistence and legacy-reader removal mostly offset
                      the full dependency-graph upgrade (serde, chrono,
                      rayon, csv, and friends), for a net growth of 33,392
                      bytes (0.18%).


      - 0.14.4:       18,973,840 bytes (≈18.1 MB). Deterministic
                      embedding and timeseries persistence added sorted-map
                      serialization paths, for a net growth of 33,072 bytes
                      (0.17%).
      - 0.14.5:       19,056,688 bytes (≈18.2 MB). +82,848 bytes (0.44%)
                      from the disk arena-guard hardening (owned
                      DiskQueryGuard acquired across mutation, algorithm,
                      and wrapper read paths), the stacker on-demand-stack
                      dependency in the Cypher parser, and the generic
                      WorkspaceGraphHooks lifecycle replacing CodeTreeHooks.


      - 0.15.1:       19,536,400 bytes — **unchanged** from 0.15.0. The
                      release is fixes plus manifest/CI metadata: the disk
                      id-index writer resolving names instead of manufacturing
                      keys, the `mcp-methods` minimum, MSRV declarations, and a
                      test-only helper regated behind `cfg(test)`. Nothing that
                      moves code size. Recorded rather than omitted so the
                      ledger has a row per release — an absent row reads as an
                      unmeasured release, not an unchanged one.

      - 0.15.0:       19,536,400 bytes (≈18.6 MB), +479,712 (+2.5%) over
                      0.14.5. Four features and two fixes account for it:
                      the three-rung durability level plus its on-demand
                      barrier and one-write frame append; the single-writer
                      lease with its `.lock-owner` sidecar and errno-based
                      contention classification; the typed constraint/conflict
                      error taxonomy; locked-schema node-label validation
                      (`schema_check.rs` +270 lines, the largest single
                      contributor); position-preserving bucket undo for
                      property/range/composite indexes (`indexes.rs` +245);
                      and `UndoEntry::ColumnarHandles`. No dependency was
                      added — this is all first-party code.


      - 0.15.2:       19,536,400 bytes — **unchanged** from the prior baseline; this release moved no code size.


      - 0.15.3:       19,536,416 bytes (≈18.6 MB). +16 bytes vs 0.15.2 — the version string in the binary went
        0.15.2 -> 0.15.3 and the embedded build metadata shifted with it.
        No code change: this release is CI/test hardening, a workflow fix,
        and dependency-floor bumps, none of which reach the engine.


      - 0.15.4:       19,552,912 bytes (≈18.6 MB). +16,496 bytes: MappedGraph
                      gained an undo journal — six GraphWrite capture sites, four
                      shared capture helpers, and a node_weight_mut_silent override,
                      mirroring MemoryGraph's. Plus DirGraph::checkpoint_lsn and its
                      serde threading for the WAL replay gate. All engine code; no
                      new dependency and no new feature.


      - 0.15.5:       19,602,720 bytes (≈18.7 MB). +49,808 bytes, essentially
                      all of it the mcp-methods 0.4.2 -> 0.4.3 bump: the default
                      extension links the MCP server, which links mcp-methods,
                      and 0.4.3 adds roots adoption plus sandbox containment.
                      kglite's own contribution is ~40 lines of manifest-key
                      plumbing in kglite-mcp-server.


      - 0.15.6:       19,619,248 bytes (≈18.7 MB). +16,528 bytes (0.08%)
                      from the specialized HNSW/vector traversal paths and
                      the clustering, coreness, degree-read, and bulk-update
                      fast paths. No dependency was added.


      - 0.15.7:       19,735,424 bytes (≈18.8 MB). +116,176 bytes (0.59%),
                      reflecting the mcp-methods 0.4.4 / rmcp 3.1.1 refresh in
                      the MCP server linked into the default extension.


      - 0.15.8:       20,000,192 bytes (≈19.1 MB). Flat versus 0.15.7: the
                      recipe-query catalog, storage-mode metadata/conversion,
                      and the plan-cache mutation bypass added no measurable
                      binary size, and the MCP server module split was a pure
                      move.


      - 0.15.9:       20,116,112 bytes (≈19.2 MB). Byte-identical to 0.15.8:
                      the column-store ownership rewrite, the held-view
                      fork-to-overlay, the C-ABI embedder surface, and the
                      per-type scan memo replaced code rather than adding it,
                      and the Java wrapper ships outside the wheel.


      - 0.15.10:       20,116,128 bytes (≈19.2 MB). +16 bytes over 0.15.9:
                      the expression-semantics hardening (checked arithmetic,
                      duplicate-column detection, backtick escaping) and the
                      add_connections interned-key rewrite replaced code
                      rather than adding it; the Java DSL ships outside the
                      wheel entirely.


      - 0.15.11:       20,116,144 bytes (≈19.2 MB). +16 bytes over 0.15.10:
                      the embedding ingest primitive (`kglite::api::embeddings`),
                      the four C-ABI embedding symbols, and the raw-query-vector
                      `text_score` path replaced code rather than adding bulk;
                      the Java surface ships outside the wheel entirely.


      - 0.15.12:       20,116,144 bytes — **unchanged** from the prior baseline; this release moved no code size.


      - 0.15.13:       20,099,632 bytes (≈19.2 MB) — **shrank** 16 KB: the
                      describe scan dropped its per-row owned-pair
                      materialization and the dead second schema pass left
                      row_properties; the planner/index alias fixes and the
                      disk arena rework are size-neutral in the wheel (the
                      disk backend's query_arena is small and the deleted
                      arena fields offset it).


      - 0.15.14:       20,182,352 bytes (≈19.2 MB) — +66 KB over 0.15.13:
                      durable sessions (session/durable.rs + the shared
                      durability module + save_guard), the membership set,
                      the unified ordering module, and LayeredRangeIndex;
                      partially offset by the four deleted ordering
                      implementations and the deduped conflict-mode parse.


      - 0.16.0:       20,264,992 bytes (≈19.3 MB). Grew with the shape-convergence +
        perf programs: cell-grained journal entries, v6 delta-varint codec +
        v5 read-compat, compiled scan-eval routes, per-column Arc store,
        NodeRemap; the Compact deletion offset most of it.


      - 0.16.1:       20,364,208 bytes (≈19.4 MB). Grew ~97 KB with the backlog-zero
        train: dynamic-label resolution, the column filter + column-major
        stats, vector scalar fns, schema_json, the lease/_ex and
        define_schema C symbols, and the magic/io split.


      - 0.16.2:       20,464,160 bytes (≈19.5 MB). +100 KB over 0.16.1,
        almost all from the Neo4j-client compatibility surface (Bolt
        dbms.*/SHOW DATABASES intercepts, elementId + its slot-anchor
        planner pass, apoc.meta/db.schema shims, EXPLAIN plan metadata);
        smaller additions from the Java query-options C-ABI binding
        (storage-format-version + execute_*_opts).


      - 0.16.3:       20,497,376 bytes (≈19.5 MB). +32 KB over 0.16.2,
        all in the bundled MCP server: the graph-mode refresh machinery
        (reload_graph tool, graph_reload state, the graph_watch watcher
        arm, since retired in favour of per-call stat freshness), the
        extensions.tools_allow allowlist
        pass, the code-tool gating predicate, and the mcp-methods
        0.4.4 -> 0.4.5 bump (builtins.github opt-in gate).


      - 0.16.4:       20,729,056 bytes (≈19.8 MB). Property-type + structural-field
      constraint enforcement, the CDC v1 subsystem (capture markers, ring
      log, db.cdc.* procedures), and the parallel runtime (dedicated pool,
      partitioned scan/aggregate paths, per-thread regex copies).


      - 0.16.5:       20,811,728 bytes (≈19.8 MB). Relationship existence/type
      constraints (entity-aware gates at four edge write paths), CDC v2
      (before-image capture, selectors, epoch handoff), and the PropMap
      property container with its serde impl.


      - 0.16.6:       20,944,128 bytes (≈20.0 MB). Two response programs: .kgl section digests + zstd
        checksums, the bidirectional BFS engine + shortest-path family
        filters/direction/single-source, var-length trail-soundness gate +
        lowering pass + expansion budget, write_scope full-verb coverage,
        MCP operator scope + isError adoption (mcp-methods 0.4.6), and the
        diagnostics-to-every-surface plumbing.


      - 0.16.7:       20,977,280 bytes (≈20.0 MB). The eval-response double
        (PR #130): the fifth warning family (type-mismatch, declared types),
        lock_schema promotion, store-uniqueness vector routing + embedding
        pruning on delete, networkx type_id round-trip, warning-policy knob.


      - 0.16.8:       20,977,264 bytes (≈20.0 MB). Comment-audit fix program — net 16 bytes
        smaller: the fix set (pk presence gate, wrapper-aware routing, mapped
        invalidation hooks) is offset by deleted dead state (metadata_dirty,
        fork edge overlay, residue blocks). No dependency changes.


      - 0.16.9:       20,977,248 bytes (≈20.0 MB). Digest verification moved to crc32fast
        (hardware CRC dispatch) and the load-throughput cells landed — the
        crc32fast dependency edge is the only growth; no feature changes.


      - 0.16.10:       21,060,032 bytes (≈20.1 MB). 0.16.9 -> 0.16.10: the lexical retrieval lane (TextIndex + BM25
        kernel + analyzer + FusedTextBm25TopK operator), the IndexFreshness
        framework, KGLTIDX1 persistence, text_bm25/score_fuse scalars, and
        the vector-index catch-up path.


      - 0.16.11:       21,473,664 bytes (≈20.5 MB). 0.16.10 -> 0.16.11: the ontology layer
        (declaration store, materialization + managed-label closures,
        ontology_audit + declaration-driven validators, blueprint gate),
        label alternation (n:A|B), and structured data (table property
        registry, declared shapes, nested SET paths, table.upsert/delete).


      - 0.16.12:       21,490,192 bytes (≈20.5 MB). 0.16.11 -> 0.16.12: the two new
        ontology audit checks (required_properties/property_types scanners),
        per-check enforcement severities, and inverse opt-in plumbing.


      - 0.16.13:       21,556,304 bytes (≈20.6 MB). +66 KB: ontology exempt
        map + audit breakdown + edge_property_violation proc, alternation
        count fusion + per-branch probes, ClosureProbe EXPLAIN rows.


      - 0.16.14:       21,572,880 bytes (≈20.6 MB). +16 KB: connectivity
        poison-distrust guards (load + disk sidecar), persist-boundary
        determinism sorts, 14 api facade re-exports, and the longer
        max_work_units budget error text.


      - 0.16.15:       21,605,984 bytes (≈20.6 MB). +33 KB: row_limit cap
        machinery + truncation diagnostics, LoadOptions + storage-override
        load path, the load-memory estimator + ceiling with the
        LoadMemoryLimit error code, spill janitor (both prefixes) and
        loader error-context wrapping.


      - 0.16.16:       21,605,984 bytes — **unchanged** from the prior baseline; this release moved no code size.


      - 0.16.17:       21,605,984 bytes — **unchanged** from the prior baseline; this release moved no code size.


      - 0.16.18:       21,622,576 bytes (≈20.6 MB). +16 KB: MCP
        peripheral-boot degradation — CsvHttpState (auto-port bind +
        failure carrier), per-root source-root resolution with
        SourceRootStatus, selftest stderr-tail capture, bundled-override
        mode-gate tolerance, and the mcp-methods 0.4.6→0.4.7 bump.


      - 0.16.19:       21,572,880 bytes (≈20.6 MB). Shrank 49,696 B: the lazy
        writer lease (`WriteOwnership`, labeled owner record), stat-per-call
        freshness and the disk-directory lease/identity work added less than
        the retired `graph_watch` watcher arm and its dormancy state removed.


      - 0.16.20:       21,605,968 bytes (≈20.6 MB). Grew 33,088 B: the
        `GraphFileIdentity::modified` accessor, the writer lease's owner-path
        field and release-record append, and the fastembed 5 -> 6 dependency
        tree; the MCP-server changes (manifest writable key, clean-save
        no-op, load/file-saved identity) live in the server binary, not here.


      - 0.16.21:       21,606,016 bytes (≈20.6 MB). +48 bytes over 0.16.20:
        the typed `KgError::CypherTimeout` (message field + real
        elapsed/limit figures, shared executor interrupt check) and the
        from_records unknown-key guard with its did-you-mean strings; the
        MCP write-gate work (force opt-in, read_only pin, UTC owner record)
        lives in the server binary, not here.


      - 0.16.22:       21,688,704 bytes (≈20.7 MB, +82,688 over 0.16.21). The
        blueprint loader gained FK-edge properties, node labels, a list column
        type, union-target junction edges and a serde catch-all on five spec
        structs for unknown-key reporting; the ontology audit gained a
        per-property census. No new dependency.


      - 0.16.23:       22,168,560 bytes (≈21.1 MB). +480 KB: the blueprint
                       input layer (Source seam, delimited + frame readers, the
                       type pre-pass) and the `xlsx` reader — calamine with
                       zip + quick-xml, the release's one new dependency
                       subtree (13 packages, feature-gated in the core crate,
                       always on in the wheel).
      - 0.16.24:       22,515,888 bytes (≈21.5 MB), +347,328 bytes (+1.6%).
                       Adds retrieval diagnostics/policy, node-property
                       ontology contracts, complete WAL state capture and
                       correctness checks across queries, blueprints and exports.


      - 0.17.0:       23,127,920 bytes (≈22.1 MB), +612,032 bytes (+2.72%).
                      The cumulative contract work adds strict raw-query
                      admission, stored-reference normalization, outline
                      identity handling, and precision-preserving protocol
                      output. This aggregate artifact measurement does not
                      isolate their individual contributions.


      - 0.17.1:       23,359,504 bytes (≈22.3 MB). +231,584 (+1.0%) over 0.17.0: eleven
                  new WAL op families with their replay/fold arms (identity aliases,
                  schema/ontology/index/constraint/spatial declarations, timeseries
                  and embedding payloads), the disk index-freshness gate and scan
                  fallbacks, three planner passes (WITH-WHERE hoist, aliasing-WITH
                  fold, terminal-RETURN hoist), cross-type/temporal comparison arms,
                  and the DescribeRequest/DescribeSurface split.


      - 0.17.2:       23,425,648 bytes (≈22.3 MB). +66,144 bytes (+0.28%)
                      over 0.17.1, encompassing the new CALL scope/import-set
                      execution, Cypher 25 clause parsing/execution, and the
                      planner's scope context. No dependency was added; this
                      aggregate artifact measurement does not isolate their
                      individual contributions.


      - 0.17.3:       23,425,664 bytes — query-fusion binding/score repairs and mapped COUNT admission;
                       16 bytes above the prior recorded baseline.


      - 0.17.4:       23,789,568 bytes (≈22.7 MB). +363,904 bytes (+1.55%)
                      over 0.17.3, encompassing canonical bounded agent
                      responses, retained-result navigation and expansion,
                      the CLI response cache, and JSONL agent controls.


      - 0.17.5:       23,789,568 bytes — **unchanged** from the prior baseline; this release moved no code size.


      - 0.17.6:       24,186,992 bytes (≈23.1 MB). +397,424 bytes (+1.67%)
                      over 0.17.5. The measured artifact is the wheel cdylib, so
                      the CLI `skill` subcommand is outside it; what grew is
                      engine-side: `kglite::api::recipes` — the catalogue model,
                      the JSON-schema keyword allowlist and the validation pass
                      (the `$param` ↔ `properties`/`required` match and the
                      read-only Cypher parse gate), lifted out of
                      kglite-mcp-server so the wheel carries them for the first
                      time — plus `kglite::api::skills` (records, frontmatter
                      render/parse, import/export), the graph-carried
                      `KgliteSkill`/`KgliteRecipe` record layers, the describe
                      skills/recipes section, and the two new pyapi modules
                      backing twelve Python methods. No dependency was added;
                      this aggregate artifact measurement does not isolate their
                      individual contributions.


      - 0.17.7:       24,203,584 bytes — +16,592 bytes (+0.07%) over 0.17.6.
                      The producer methodology layer in `kglite-mcp-server`,
                      which the default extension links: `ServerExtensions`
                      gains two builders and two stored fields
                      (`with_skills`/`with_recipes`), `skills.rs` gains the
                      producer layer plus the extracted `boot_skills`, a new
                      `tools/activation_refresh.rs` wraps `set_root_dir` /
                      `repo_management` to re-resolve the registry, and three
                      types are re-exported from the crate root. No engine
                      code changed, which is why the delta is four orders of
                      magnitude smaller than 0.17.6's.


      - 0.17.8:       24,733,008 bytes (≈23.6 MB). +530 KB: the Obsidian vault
        dialect on okf (profile, attachments, vault.yaml, validator), the vault
        exporter with manifest and YAML emitter, fingerprint/provenance/rebuild,
        api::embeddings::embed_property, and the fused-aggregate label filters.


      - 0.17.9:       25,245,584 bytes (≈24.1 MB). +513 KB: the vault structure
        profile (pulldown-cmark block tree, section/chunk/callout/list/table
        derivation, edge tables on import and export, key_from_heading) and
        the re-seated link scanner.


      - 0.17.10:      25,245,584 bytes — unchanged to the byte: the two Cypher
        executor fixes (WITH scope barrier, mixed-`*` projection; ~230 lines)
        fit inside the existing page-aligned `__TEXT` segment.


      - 0.17.11:       25,328,256 bytes (≈24.2 MB, +82,672 / +0.3%): the vault
        annotation loader — `okf::directives` (section properties and typed
        edges from `<!-- kglite … -->`), `okf::tags` (`tag_labels:` rules and
        per-node `tags`), synthetic-heading promotion in the block tree, typed
        inline links and over-cap block splitting in the chunker — plus the
        recipe `default:` binding and the dialect stamp in the `.kgl` header.


      - 0.17.12:       25,411,008 bytes macOS (≈24.2 MB, +82,752 / +0.3%): `okf::cache`
        (`open`, `CachePolicy`, `Opened`, the cache-artifact predicate and two
        provenance stamps), the named recipe tool routes and the catalogue
        budget renderer, plus the `tool` field carried through the recipe
        record, its Python binding and the CLI `okf open` subcommand.
        The published manylinux2014 x86_64 abi3 wheel's stripped
        `kglite/kglite.abi3.so` is 31,188,600 bytes (wheel SHA-256
        `9ac59f03e81dea8bd59d87f68057f8f4113529c3bb29a1ff1baf27bf94196b53`).
        CI measures the bare Ubuntu host's pre-wheel `target/release` output,
        so linker/glibc and auditwheel differences prevent byte identity; the
        released member replaces the prior unverified Linux estimate as the
        closest reproducible user-artifact baseline.


      - 0.18.0:       26,337,920 bytes (≈25.1 MB). +926,912 (+3.6%) over 0.17.12: the
        relationship-embedding programs — relationship vector stores, HNSW and BM25
        indexes with .kgl/.kgle persistence, the node and relationship embedding
        procedure namespaces and their routers, the Python/Rust relationship writers,
        cross-type retrieval, the fused null-filter and WITH-shaped top-k routes, and
        the bound-value planner anchors.


      - 0.18.1:       26,354,464 bytes (≈25.1 MB). +16,544 bytes: path-binding
        helpers, the load-time reference-candidate scan, tagged date/duration
        JSON parameters and the parser's widened value positions.


      - 0.19.0:       27,165,040 bytes (≈25.9 MB). +810,576 bytes: the
        valid-time programs — declared intervals with the endpoint index
        and guard templates, the core FluentFilter, filtered HNSW search,
        write-time interval checks and write-clause folding, the
        per-statement temporal diagnostics, the bulk subgraph gather, and
        the disk overlay flush with typed column flattening.


      - 0.19.1:       27,793,568 bytes (≈26.5 MB). +628,528 bytes: the
        register-scale disk build (typed timestamp column, column tail
        store, disk cells journal, hard-linked incremental publish, carried
        title/id bundles, stale-save refusal), the export manifest, lossless
        CSV export, and the RDF 1.2 importer and N-Quads/TriG exporter.
        Linux: the release refresh writes only the host platform's entry,
        so the Linux baseline had stayed at the published 0.17.12 member
        through 0.18.0-0.19.1 while the published members grew to
        32,370,552 (0.18.0), 32,383,416 (0.18.1), 33,352,888 (0.19.0) and
        34,169,144 (0.19.1) — 0.19.1 already sat at +9.6% of the old gate.
        The Linux baseline is now the published 0.19.1 manylinux2014 x86_64
        member `kglite/kglite.abi3.so`, 34,169,144 bytes (wheel SHA-256
        `26c75d0f53fc3a753a3201ffeda7aa06f677c41611c0ad47cd44a1dd79f7e2ed`).

      - 0.19.2:       28,041,872 bytes (≈26.7 MB), +248,304 (+0.9%) over
        0.19.1: the valid-time default context, chain path/distinct/grouped
        count operators, streaming match aggregation, the per-instant UNWIND
        valid_at join, the cached peer-count histogram and the identical-row
        tracker.

      - 0.19.3:       28,207,248 bytes (≈26.9 MB), +165,376 (+0.6%) over
        0.19.2: the parent-link resolver, grouped build diagnostics with
        strict mode, the missing-endpoint policy, load-time data advisories,
        the stored valid-time default, valid_instant(), calendar date
        arithmetic, mask-based node counts and correlated lookup probes.

      - 0.19.4:       28,223,840 bytes (≈26.9 MB), +16,592 (+0.06%) over
        0.19.3: free-list slot allocation for the forked overlay, the
        panic-safe fold, the heap-store append tail, Int64 widening of
        unique-id columns, numeric WAL replay identity and valid-time
        checks on every property writer.

      - 0.19.5:       28,290,096 bytes (≈27.0 MB), +66,256 (+0.2%) over
        0.19.4: delete compaction on commit, streaming WAL replay and the
        power-loss quarantine/torn-tail saves, MERGE returning every match
        with the unique-tuple probe, the WITH … LIMIT pushdown shapes,
        path-variable rebinding checks, unaliased column display names and
        the name/title index routing.

      - 0.19.6:       28,571,264 bytes (≈27.2 MB), +281,168 (+1.0%) over
        0.19.5: online backup and online/automatic checkpoints, write-time
        ontology enforcement (statement and transaction-end judges, bulk
        frame gates), the fork overlay's adjacency layer and op-log
        fold-back, in-place auto-commit with its undo path, the batch read
        cursor, the per-query cancel token, lazy property NDV and the
        lifted durable-open / auto-commit session API.

      - 0.19.7:       28,868,960 bytes (≈27.5 MB), +297,696 (+1.0%) over
        0.19.6: the core group-commit queue with fsync outside the graph
        lock and publish-after-durable, aggregate window functions, the
        temporal TCK cluster, WAL-recorded text indexes, the non-aggregating
        WITH scope rule, the per-value parameterised plan cache and the
        UNWIND/WITH key pushdown into patterns.

      - 0.20.0:       30,289,904 bytes (≈28.9 MB), +1,420,944 (+4.9%) over
        0.19.7: the temporal value model (local and zoned times, zoned
        datetimes with fixed offsets and IANA zones through chrono-tz's
        embedded tz database, nanosecond durations, their parsers, accessors
        and arithmetic), the .kgl v8 header checksum, the durable Python
        `open_session` and its session classes, and the per-statement clock.

      - 0.20.1:       30,339,312 bytes (≈28.9 MB), +49,408 (+0.2%) over
        0.20.0: the mapped fork overlay base (`ForkBase`), the shared lazy
        id-index slot and per-type generations, hand-written `NodeData` /
        `EdgeData` serde, and the `thin-vec` container.

      - 0.21.0:       31,382,432 bytes (≈29.9 MB), +1,043,120 (+3.4%) over
        0.20.1: the .kgl v9 mappable sections (raw columns, topology, type
        and id-index sections), the frozen mapped topology backend and its
        shared paged write overlay, rebase on save, the unsaved-write
        budget, embedding quantization, and the MCP stale-build reload.

      - 0.21.1:       31,845,520 bytes (≈30.4 MB, +463 KB). Owned
        constraint indexes, relationship property indexes and UNIQUE,
        FULLTEXT/VECTOR index DDL with property-sourced vector indexes,
        the SEARCH clause, three planner passes (null-safe seek, CALL
        hoist, WITH-boundary top-K), and the waiting borrow helpers.

      - 0.21.2:       32,043,968 bytes (≈30.6 MB, +198 KB). Ontology
        contracts (`distinct`, `property_values` with compiled regexes,
        `by_source`, label exemptions, the `distinct` drill-down and
        declare-time warnings), ranked build diagnostics, and the
        single-pass audit edge grouping.


      - 0.21.3:       32,160,448 bytes (≈30.7 MB), +116,480 (+0.36%)
        over 0.21.2. mcp-methods 0.6.1 (subscriptions/listen, schema
        standardization) with rmcp 3.8, pyo3 0.30, the full lockfile
        refresh and the Bolt handshake deadline; the deleted boltr 0.2
        guards offset part of it.


      - 0.21.4:       32,160,448 bytes, unchanged in total. `__TEXT`
        grew 16 KiB against the published 0.21.3 wheel (rustyline 19,
        mcp-methods 0.6.2) and still fit the host build's page padding.

    Raising the baseline is a deliberate act — every bump should
    be accompanied by an updated growth note above. For a precise
    drilldown, run `cargo bloat --release --crates --filter kglite`.
    """

    # Post-G.4 the wheel cdylib is the kglite-py crate's output —
    # `libkglite_py.{dylib,so}` — not the engine's `libkglite.{dylib,so}`
    # (which is now an rlib + dylib pair for downstream Rust crates).
    # The wheel artifact is what users `pip install`, so only kglite-py's
    # cdylib is a valid measurement. Falling back to the core library can make
    # this gate pass while the shipped extension is absent or oversized.
    candidates = [
        REPO_ROOT / "target" / "release" / "libkglite_py.dylib",
        REPO_ROOT / "target" / "release" / "libkglite_py.so",
        REPO_ROOT / "target" / "release" / "kglite_py.dll",
    ]
    bin_path = next((p for p in candidates if p.exists()), None)
    if bin_path is None:
        pytest.fail("kglite-py release cdylib is missing — run `maturin build --release`")

    size = bin_path.stat().st_size
    # Never fall back to another platform's number: each baseline describes one
    # platform's artifact, so comparing e.g. a Windows MSVC DLL against the
    # Linux wheel member would report a meaningless pass or failure.
    if sys.platform not in BINARY_SIZE_BASELINES:
        pytest.skip(f"no measured binary-size baseline for {sys.platform}; capture one before gating it")
    platform_key = sys.platform
    baseline, baseline_version = BINARY_SIZE_BASELINES[platform_key]
    gate = int(baseline * 1.10)
    assert size <= gate, (
        f"{bin_path.name} = {size:,} bytes > gate {gate:,} "
        f"(+10% over {baseline_version} {platform_key} baseline {baseline:,}). "
        "Investigate what grew before raising the gate — see the "
        "growth note in this test's docstring for the breakdown shape."
    )
