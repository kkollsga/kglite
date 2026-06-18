# Feedback: workspace `repo_management` listing reads like a tombstone wall, not an "analyzed repos" catalog

**From**: MCP-servers operator
**To**: kglite team
**Date**: 2026-06-18
**Re**: `mcp-methods` workspace inventory — sweep leaves cruft, `list()` framing alarms instead of cataloguing

## Symptom

After months of use, the GitHub workspace server (`--workspace .../open_source`)
lists **11 live repos + 104 "STALE — re-fetch with repo_management('…')" lines**.
The stale block reads like a wall of dead tombstones / warnings, when what the
operator actually wants is a **catalogue of repos we have analyzed**, with access
counts, kept as history.

Two concrete asks from the operator:

1. Present **two lists**: (a) active/live repos, and (b) **all repos we've analyzed,
   with access counts** — framed as an archive, not as "STALE — re-fetch" alarms.
2. Stop accumulating dead artifacts that the sweep should be reclaiming.

## Root cause (mcp-methods `crates/mcp-methods/src/server/workspace.rs`)

The auto-sweep *partly* works — it reclaims the clone dirs (good; `repos/` held only
the 11 live clones) — but:

1. **Tombstones never purged.** `sweep_stale()` (L325) deletes the clone dir then only
   sets `entry.stale = true` (L354); the inventory entry lives forever. Same in
   `delete()` (L553 → `mark_stale`).
2. **`stale` is auto-derived from disk** on every load (L287-291: any entry without an
   on-disk clone is forced `stale = true`). So an operator **cannot** reframe the
   listing by editing `inventory.json` — kept entries always re-print as STALE.
3. **`list()` framing is hardcoded** (L592): every stale entry renders as
   `"  {rname}  [STALE — re-fetch with repo_management('{rname}')]"`. There is no
   "analyzed archive" view.
4. **Orphaned `.kgl` graphs never swept.** `sweep_stale()` only removes `repos/<org>/<repo>`;
   it never touches `graphs/<org>/<repo>.kgl`. We have 8 orphan graphs (~40 MB, incl. a
   37 MB `dotnet/runtime.kgl`) for repos whose clones are long gone.
5. **`delete()` docstring is wrong** (L623 claims it removes "the named repo + inventory
   entry"); the code only marks stale (L553). Either the doc or the behavior should change.

## Suggested fix

- **Reframe `list()` into two sections** keyed off `stale`, e.g.:
  - `Active repos (N):` — live clones (current behavior, keep the `[active]` marker).
  - `Analyzed (M) — clone removed, re-fetch to reopen:` — all stale entries, **sorted by
    `access_count` desc**, shown as `  org/repo  (K analyses, last <relative>)` with **no
    "STALE" / no per-line re-fetch noise**. One footer line can carry the re-fetch hint.
  This reads as a catalogue of what's been studied rather than a failure list.
- **Sweep the orphan graph too**: when `sweep_stale()` / `delete()` removes
  `repos/<org>/<repo>`, also remove `graphs/<org>/<repo>.kgl` (and prune empty org dirs
  under `graphs/`, mirroring `prune_empty_org_dirs()`).
- **Optional**: a hard-purge knob (e.g. `repo_management(prune=true)` or a much longer
  second TTL) to drop inventory entries entirely for operators who don't want history —
  but the **default should keep** the analyzed record per this request.
- **Fix the `delete()` docstring** to match "marks stale + removes clone" (or make it
  actually drop the entry, gated).

## How to reproduce

1. `--workspace DIR` (kind: github), clone several repos, let `--stale-after-days`
   (default 7) elapse without touching them.
2. Call `repo_management()` → sweep removes the clones but the listing keeps printing
   each as `[STALE — re-fetch …]`; `graphs/<org>/<repo>.kgl` remains on disk.

## Versions

- kglite / kglite-mcp-server **0.11.2** (pip wheel), venv `~/venvs/mcp-venv` (Py 3.14).
- mcp-methods crate as vendored into the 0.11.2 build.

## Severity / urgency

Low — cosmetic + minor disk leak (~40 MB here), no functional breakage. But the listing
is the operator's main view of the workspace and currently misrepresents an analyzed-repo
archive as a wall of failures.
