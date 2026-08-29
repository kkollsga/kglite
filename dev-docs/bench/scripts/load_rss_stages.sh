#!/usr/bin/env bash
# load_rss_stages.sh — three-plateau RSS profile for a `.kgl` load.
#
# WHY THREE PLATEAUS. A `.kgl` load does not settle at one number:
#
#   1. PEAK    — the high-water mark *during* decode. Transient buffers
#                (decompressed topology held across index rebuild, sized
#                decompress arenas) live only here, so this is the number a
#                machine actually has to survive; a jetsam kill happens here,
#                not at rest.
#   2. SETTLED — RSS once the process is idle after load. What the graph
#                costs to hold.
#   3. POST-Q  — RSS after the first node-touching query. `id_indices` is
#                built lazily per node type on first use
#                (`dir_graph/mod.rs`, `#[serde(skip)] pub id_indices`), so a
#                load-only measurement structurally under-reports the
#                resident cost of a graph that is actually being queried.
#
# HOW IT MEASURES. The release CLI's `session` subcommand keeps one graph
# loaded and reads JSONL requests from stdin, so the process stays alive for
# stages 2 and 3. stdin is a FIFO the script feeds one line at a time; the
# session's reply on stdout is the stage barrier (the loop only starts reading
# stdin *after* `open_or_create_graph` returns, so the reply to the first
# request is an exact load-complete marker). RSS is sampled by polling
# `ps -o rss=` at --interval ms from a background sampler.
#
# CROSS-CHECK. `/usr/bin/time -l` on a separate one-shot `kglite query` run
# reports the kernel's own maximum resident set size. It covers load + a
# trivial query in one process, so it should sit at or just above this
# script's PEAK; a large disagreement means the poller missed the spike
# (raise --interval resolution) and the number should not be trusted.
#
# RSS IS NOT WHAT JETSAM KILLS ON. macOS's memory manager judges a process by
# its *phys_footprint* — which excludes clean, file-backed pages that can be
# dropped and re-read. RSS includes them. On sodir the gap is not a rounding
# error: 227 MB RSS vs 168 MB footprint (peak footprint 235 MB), because the
# load spills ~167 MiB of columns to a temp dir and mmaps them back. So this
# script samples BOTH, and the footprint columns are the ones to quote when the
# question is "will this machine survive the load". `vmmap` costs ~1s, far too
# slow to poll, so footprint is sampled once at SETTLED and once after the first
# query; its own "Physical footprint (peak)" counter covers the load transient
# that polling would miss. Requires `vmmap` (macOS); the columns are empty
# elsewhere.
#
# PAGE-CACHE CAVEAT (jetsam). Columns >= 256KB are written to a spill file and
# mmap'd during load in BOTH `memory` and `mapped` modes
# (`io/file.rs` passes a temp dir unconditionally), so part of the graph's
# bytes are file-backed page cache rather than anonymous memory. RSS counts
# resident file-backed pages, but the OS can evict them under pressure while
# anonymous pages must be swapped or the process killed. Two consequences:
#   - A change that moves bytes anonymous -> file-backed looks like an RSS win
#     while changing nothing about the machine's real memory pressure.
#   - RSS after a load is sensitive to whether the source file and the spill
#     files are already warm in the page cache. This script reports the spill
#     root and the file size so a run can be read in that light; it does not
#     try to purge the cache (that needs root and perturbs the whole machine).
# Read deltas between modes/builds, not absolute bytes, when the difference is
# small.
#
# THE ALLOCATOR IS A FIRST-CLASS VARIABLE — larger than anything in the load
# path (measured 2026-08-29, sodir 133.6MB):
#
#     CLI  (system libmalloc, default)      850 MB max RSS
#     CLI  (MallocSpaceEfficient=1)         370 MB max RSS
#     wheel (mimalloc v2, kglite-py)        472 MB settled
#
# Same engine, same file, same release profile: a 2.3x spread that no engine
# change produced. So:
#   - Never compare a CLI number to a wheel number. They link different
#     allocators (`crates/kglite-py/src/lib.rs` sets mimalloc as the global
#     allocator; the CLI takes the platform default).
#   - `--space-efficient` sets MallocSpaceEfficient=1 for the measured process
#     (macOS libmalloc; tighter size classes). Use it to see the graph rather
#     than the allocator's padding — it is the only setting under which the
#     transient load peak is visible above the noise on this platform, and the
#     only one under which per-structure costs (e.g. index rebuild) are not
#     absorbed into pre-existing allocator slack. Note it is NOT in this
#     machine's `man malloc`; it is verified empirically here, not from docs.
#   - mimalloc is NOT purge-sensitive here: MIMALLOC_PURGE_DELAY of -1, 0 and
#     default all gave 471-472 MB on sodir, so the wheel's number is retained
#     live memory, not deferred purging.
#
# STORAGE MODES. The CLI opens graphs in `StorageMode::Memory` only (see
# `crates/kglite-cli/src/lib.rs`) — there is no `--storage` flag. `mapped` is
# therefore measured with `--driver python`, which uses
# `kglite.open(path, storage="mapped")` in a Python process. THE TWO DRIVERS
# ARE NOT COMPARABLE: the Python driver carries interpreter + wheel-import
# overhead (tens of MB) on top of the graph. Compare python-vs-python and
# cli-vs-cli only; every output row records its driver for exactly this reason.
#
# USAGE
#   load_rss_stages.sh --graph FILE.kgl [options]
#
#     --graph PATH        .kgl to load (required)
#     --mode MODE         memory | mapped        (default: memory)
#     --driver DRIVER     cli | python | auto    (default: auto —
#                         cli for memory, python for mapped)
#     --label NAME        row label in the CSV/summary (default: graph basename)
#     --settle-ms MS      idle wait before the SETTLED sample   (default: 3000)
#     --interval-ms MS    RSS poll interval                     (default: 25)
#     --query CYPHER      the first query, run after the SETTLED sample.
#                         (default: MATCH (n) RETURN count(n) AS c)
#                         NOTE: the default does NOT move the third plateau —
#                         a full scan does not build `id_indices`. Pass a point
#                         lookup to see that term: on indexed_500k,
#                         `MATCH (n:Item {id: 'item-000000042'}) RETURN n.uid`
#                         moved settled 325.6 -> 364.4 MB (+38.8, +12%), while
#                         the default count query moved it +0.8 MB.
#     --repeat N          run the whole measurement N times     (default: 1)
#     --csv PATH          append rows here (default: dev-docs/bench/out/load_rss.csv)
#     --bin PATH          kglite binary (default: target/release/kglite)
#     --python PATH       python for --driver python (default: repo .venv)
#     --space-efficient   run the measured process with MallocSpaceEfficient=1
#                         (see "THE ALLOCATOR IS A FIRST-CLASS VARIABLE")
#     --timing            set KGLITE_LOAD_TIMING=1 and echo the [TIMING] lines.
#                         NOTE: the stage timers are instrumented in
#                         `load_disk_dir` only, so a `.kgl` load emits nothing
#                         — this flag is useful for disk-directory graphs.
#     --no-time-l         skip the /usr/bin/time -l cross-check run
#
# OUTPUT. A human-readable block per run on stdout, plus one CSV row per run
# appended to --csv with columns:
#   ts,label,graph,file_bytes,mode,driver,alloc,run,peak_kb,settled_kb,postq_kb,
#   fp_settled_kb,fp_postq_kb,fp_peak_kb,load_ms,query_ms,time_l_max_rss_kb,
#   peak_over_settled,settled_over_file
# The `peak_kb`/`settled_kb`/`postq_kb` trio is RSS; the `fp_*` trio is physical
# footprint (`fp_peak_kb` is the kernel's own peak, so it cannot miss a spike).
# `alloc` is `default` or `spaceeff` — a row is only comparable to rows sharing
# its driver AND its alloc.
#
# ARTIFACTS. Everything this script writes goes to dev-docs/bench/out/ (CSV,
# transient FIFO/stdout captures under a mktemp dir that is trapped clean).
# Nothing is written next to the graph or into results/ — the Python driver
# passes durable="off", lock=False for exactly that reason; the default `open`
# leaves a `<graph>-wal` and `<graph>.lock-owner` beside the fixture.
#
# Release profile only — a debug-built binary's numbers are invalid here
# (CLAUDE.md, "Performance protocol").

set -euo pipefail

# A comma decimal separator (any non-C LC_NUMERIC — this machine defaults to
# nb_NO) makes awk emit "1,02", which silently splits the CSV row into extra
# fields. Pin the numeric locale for the whole script.
export LC_ALL=C

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
OUT_DIR="$REPO_ROOT/dev-docs/bench/out"

GRAPH=""
MODE="memory"
DRIVER="auto"
LABEL=""
SETTLE_MS=3000
INTERVAL_MS=25
QUERY='MATCH (n) RETURN count(n) AS c'
REPEAT=1
CSV=""
BIN="$REPO_ROOT/target/release/kglite"
PYTHON="$REPO_ROOT/.venv/bin/python"
TIMING=0
TIME_L=1
SPACE_EFFICIENT=0

die() { echo "load_rss_stages.sh: $*" >&2; exit 2; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --graph)       GRAPH="$2"; shift 2 ;;
    --mode)        MODE="$2"; shift 2 ;;
    --driver)      DRIVER="$2"; shift 2 ;;
    --label)       LABEL="$2"; shift 2 ;;
    --settle-ms)   SETTLE_MS="$2"; shift 2 ;;
    --interval-ms) INTERVAL_MS="$2"; shift 2 ;;
    --query)       QUERY="$2"; shift 2 ;;
    --repeat)      REPEAT="$2"; shift 2 ;;
    --csv)         CSV="$2"; shift 2 ;;
    --bin)         BIN="$2"; shift 2 ;;
    --python)      PYTHON="$2"; shift 2 ;;
    --space-efficient) SPACE_EFFICIENT=1; shift ;;
    --timing)      TIMING=1; shift ;;
    --no-time-l)   TIME_L=0; shift ;;
    -h|--help)     sed -n '2,90p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *)             die "unknown argument: $1" ;;
  esac
done

[[ -n "$GRAPH" ]] || die "--graph is required"
[[ -f "$GRAPH" ]] || die "no such graph: $GRAPH"
GRAPH="$(cd "$(dirname "$GRAPH")" && pwd)/$(basename "$GRAPH")"
[[ -n "$LABEL" ]] || LABEL="$(basename "$GRAPH" .kgl)"
[[ -n "$CSV" ]] || CSV="$OUT_DIR/load_rss.csv"

case "$MODE" in memory|mapped) ;; *) die "--mode must be memory|mapped" ;; esac
if [[ "$DRIVER" == "auto" ]]; then
  # The CLI has no --storage flag; mapped can only be requested from Python.
  if [[ "$MODE" == "mapped" ]]; then DRIVER="python"; else DRIVER="cli"; fi
fi
case "$DRIVER" in
  cli)
    [[ -x "$BIN" ]] || die "release CLI not found at $BIN (cargo build -p kglite-cli --release)"
    [[ "$MODE" == "memory" ]] || die "--driver cli cannot request storage=$MODE; use --driver python"
    ;;
  python)
    [[ -x "$PYTHON" ]] || die "python not found at $PYTHON"
    ;;
  *) die "--driver must be cli|python|auto" ;;
esac

mkdir -p "$OUT_DIR"
WORK="$(mktemp -d "$OUT_DIR/load_rss.XXXXXX")"
cleanup() {
  [[ -n "${SAMPLER_PID:-}" ]] && kill "$SAMPLER_PID" 2>/dev/null || true
  [[ -n "${CHILD_PID:-}" ]]   && kill "$CHILD_PID"   2>/dev/null || true
  rm -rf "$WORK"
}
trap cleanup EXIT

FILE_BYTES="$(stat -f %z "$GRAPH" 2>/dev/null || stat -c %s "$GRAPH")"

if [[ ! -f "$CSV" ]]; then
  echo "ts,label,graph,file_bytes,mode,driver,alloc,run,peak_kb,settled_kb,postq_kb,fp_settled_kb,fp_postq_kb,fp_peak_kb,load_ms,query_ms,time_l_max_rss_kb,peak_over_settled,settled_over_file" > "$CSV"
fi

# The Python session driver. Mirrors the CLI session's contract exactly — read
# a JSONL request from stdin, reply one JSON line on stdout — so both drivers
# hit the same stage barriers and the same sampling code below.
cat > "$WORK/py_session.py" <<'PYEOF'
import json, sys, time
import kglite

path, mode = sys.argv[1], sys.argv[2]
t0 = time.perf_counter()
# durable="off" + lock=False: without them `open` attaches a write-ahead log and
# a lock-owner file NEXT TO THE GRAPH, which (a) writes into the fixture's
# directory — including read-only fixtures — and (b) puts WAL machinery inside
# the number being measured.
g = kglite.open(path, storage=(None if mode == "memory" else mode),
                durable="off", lock=False)
load_ms = (time.perf_counter() - t0) * 1000.0
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    req = json.loads(line)
    op = req.get("op", "query")
    if op in ("exit", "quit"):
        print(json.dumps({"ok": True, "op": op}), flush=True)
        break
    if op == "help":
        print(json.dumps({"ok": True, "op": "help", "load_ms": load_ms}), flush=True)
        continue
    t = time.perf_counter()
    rows = g.cypher(req["query"])
    n = len(rows) if rows is not None else 0
    print(json.dumps({"ok": True, "op": "query",
                      "query_ms": (time.perf_counter() - t) * 1000.0,
                      "rows": n}), flush=True)
PYEOF

now_ms() { python3 -c 'import time;print(int(time.time()*1000))'; }

sample_kb() { ps -o rss= -p "$1" 2>/dev/null | tr -d ' '; }

# Physical footprint in KB: "<cur> <peak>". macOS only; empty pair elsewhere.
sample_footprint_kb() {
  command -v vmmap > /dev/null 2>&1 || { echo " "; return; }
  vmmap -summary "$1" 2>/dev/null | awk '
    /^Physical footprint:/        { cur  = $3 }
    /^Physical footprint \(peak\):/ { peak = $4 }
    END {
      print to_kb(cur), to_kb(peak)
    }
    function to_kb(v,   n, u) {
      if (v == "") return ""
      u = substr(v, length(v), 1); n = substr(v, 1, length(v) - 1) + 0
      if (u == "G") return int(n * 1048576)
      if (u == "M") return int(n * 1024)
      if (u == "K") return int(n)
      return int(n / 1024)
    }'
}

run_once() {
  local run="$1"
  local fifo="$WORK/in.$run" outf="$WORK/out.$run" errf="$WORK/err.$run" peakf="$WORK/peak.$run"
  rm -f "$fifo"; mkfifo "$fifo"
  : > "$outf"; : > "$errf"; echo 0 > "$peakf"

  local -a env_pfx=(env)
  [[ "$TIMING" == "1" ]] && env_pfx+=(KGLITE_LOAD_TIMING=1)
  [[ "$SPACE_EFFICIENT" == "1" ]] && env_pfx+=(MallocSpaceEfficient=1)

  local t_start t_ready t_settled t_postq
  t_start="$(now_ms)"

  if [[ "$DRIVER" == "cli" ]]; then
    "${env_pfx[@]}" "$BIN" session "$GRAPH" --format json < "$fifo" > "$outf" 2> "$errf" &
  else
    "${env_pfx[@]}" "$PYTHON" "$WORK/py_session.py" "$GRAPH" "$MODE" < "$fifo" > "$outf" 2> "$errf" &
  fi
  CHILD_PID=$!
  # Hold the FIFO open for the whole run; closing it would EOF the session.
  # Fixed fd 9, not a {var} redirect — macOS ships bash 3.2, which has no
  # varname file descriptors.
  exec 9> "$fifo"

  # Sampler: poll RSS into peakf. Started before the first request is served,
  # so it is running through the whole decode — that is where PEAK lives.
  (
    local_peak=0
    while kill -0 "$CHILD_PID" 2>/dev/null; do
      cur="$(sample_kb "$CHILD_PID")"
      if [[ -n "$cur" && "$cur" -gt "$local_peak" ]]; then
        local_peak="$cur"; echo "$local_peak" > "$peakf"
      fi
      perl -e "select(undef,undef,undef,$INTERVAL_MS/1000)" 2>/dev/null || sleep 0.05
    done
  ) &
  SAMPLER_PID=$!

  # Stage 1: load. The session reads stdin only after the graph is open, so
  # the reply to this request marks load-complete.
  echo '{"op":"help","id":"ready"}' >&9
  local deadline=$(( $(now_ms) + 900000 ))
  while [[ "$(wc -l < "$outf" | tr -d ' ')" -lt 1 ]]; do
    kill -0 "$CHILD_PID" 2>/dev/null || { echo "session died during load:" >&2; cat "$errf" >&2; return 1; }
    [[ "$(now_ms)" -lt "$deadline" ]] || { echo "timeout waiting for load" >&2; return 1; }
    perl -e 'select(undef,undef,undef,0.01)'
  done
  t_ready="$(now_ms)"
  local peak_kb; peak_kb="$(cat "$peakf")"

  # Stage 2: settle, then sample resident cost at rest.
  perl -e "select(undef,undef,undef,$SETTLE_MS/1000)"
  local settled_kb; settled_kb="$(sample_kb "$CHILD_PID")"
  local fp_settled fp_peak_settled
  read -r fp_settled fp_peak_settled <<< "$(sample_footprint_kb "$CHILD_PID")"
  t_settled="$(now_ms)"

  # Stage 3: first node-touching query -> lazy id_indices materialize.
  echo "{\"op\":\"query\",\"query\":$(printf '%s' "$QUERY" | "${PYTHON}" -c 'import json,sys;print(json.dumps(sys.stdin.read()))' 2>/dev/null || printf '"%s"' "$QUERY"),\"id\":\"q1\"}" >&9
  deadline=$(( $(now_ms) + 900000 ))
  while [[ "$(wc -l < "$outf" | tr -d ' ')" -lt 2 ]]; do
    kill -0 "$CHILD_PID" 2>/dev/null || { echo "session died during query:" >&2; cat "$errf" >&2; return 1; }
    [[ "$(now_ms)" -lt "$deadline" ]] || { echo "timeout waiting for query" >&2; return 1; }
    perl -e 'select(undef,undef,undef,0.01)'
  done
  t_postq="$(now_ms)"
  perl -e "select(undef,undef,undef,0.5)"
  local postq_kb; postq_kb="$(sample_kb "$CHILD_PID")"
  local fp_postq fp_peak
  read -r fp_postq fp_peak <<< "$(sample_footprint_kb "$CHILD_PID")"
  local peak_all_kb; peak_all_kb="$(cat "$peakf")"

  echo '{"op":"exit"}' >&9
  exec 9>&-
  wait "$CHILD_PID" 2>/dev/null || true
  CHILD_PID=""
  kill "$SAMPLER_PID" 2>/dev/null || true
  wait "$SAMPLER_PID" 2>/dev/null || true
  SAMPLER_PID=""

  local load_ms=$(( t_ready - t_start ))
  local query_ms=$(( t_postq - t_settled ))

  # /usr/bin/time -l cross-check on a fresh one-shot process. Only meaningful
  # for the CLI driver; for python it would time the interpreter startup too,
  # which is measured all the same but must be read as such.
  local time_l_kb=""
  if [[ "$TIME_L" == "1" ]]; then
    local tlf="$WORK/timel.$run"
    if [[ "$DRIVER" == "cli" ]]; then
      /usr/bin/time -l "${env_pfx[@]}" "$BIN" query "$GRAPH" "$QUERY" > /dev/null 2> "$tlf" || true
    else
      /usr/bin/time -l "${env_pfx[@]}" "$PYTHON" -c "
import sys, kglite
g = kglite.open(sys.argv[1], storage=(None if sys.argv[2]=='memory' else sys.argv[2]),
                durable='off', lock=False)
len(g.cypher(sys.argv[3]))
" "$GRAPH" "$MODE" "$QUERY" > /dev/null 2> "$tlf" || true
    fi
    # macOS reports bytes; GNU time reports KB. Normalise to KB.
    local raw; raw="$(grep -E 'maximum resident set size' "$tlf" | tr -dc '0-9' || true)"
    if [[ -n "$raw" ]]; then
      if [[ "$(uname -s)" == "Darwin" ]]; then time_l_kb=$(( raw / 1024 )); else time_l_kb="$raw"; fi
    fi
  fi

  local ratio_ps ratio_sf
  ratio_ps="$(awk -v a="$peak_all_kb" -v b="$settled_kb" 'BEGIN{ if (b>0) printf "%.2f", a/b; else print "" }')"
  ratio_sf="$(awk -v a="$settled_kb" -v b="$FILE_BYTES" 'BEGIN{ if (b>0) printf "%.2f", (a*1024)/b; else print "" }')"

  local alloc_tag; alloc_tag=$([[ "$SPACE_EFFICIENT" == "1" ]] && echo spaceeff || echo default)
  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$LABEL" "$(basename "$GRAPH")" "$FILE_BYTES" \
    "$MODE" "$DRIVER" "$alloc_tag" "$run" "$peak_all_kb" "$settled_kb" "$postq_kb" \
    "${fp_settled:-}" "${fp_postq:-}" "${fp_peak:-}" \
    "$load_ms" "$query_ms" "$time_l_kb" "$ratio_ps" "$ratio_sf" >> "$CSV"

  echo "--- $LABEL / mode=$MODE / driver=$DRIVER / alloc=$alloc_tag / run $run ---"
  awk -v f="$FILE_BYTES" -v p="$peak_all_kb" -v pl="$peak_kb" -v s="$settled_kb" -v q="$postq_kb" \
      -v lm="$load_ms" -v qm="$query_ms" -v tl="$time_l_kb" \
      -v fs="${fp_settled:-}" -v fq="${fp_postq:-}" -v fp="${fp_peak:-}" 'BEGIN{
    printf "  file            %10.1f MB\n", f/1048576;
    printf "  PEAK (load)     %10.1f MB   (%.2fx file)\n", pl/1024, (pl*1024)/f;
    printf "  PEAK (whole)    %10.1f MB   (%.2fx file)\n", p/1024, (p*1024)/f;
    printf "  SETTLED         %10.1f MB   (%.2fx file)\n", s/1024, (s*1024)/f;
    printf "  POST-1ST-QUERY  %10.1f MB   (%.2fx file)\n", q/1024, (q*1024)/f;
    printf "  peak/settled    %10.2f\n", (s>0? p/s : 0);
    printf "  load wall       %10d ms\n", lm;
    printf "  first query     %10d ms\n", qm;
    if (tl != "") printf "  time -l maxrss  %10.1f MB   (one-shot cross-check)\n", tl/1024;
    if (fs != "") {
      printf "  footprint       %10.1f MB settled, %.1f MB post-query, %.1f MB peak\n", fs/1024, fq/1024, fp/1024;
      printf "                  (jetsam judges footprint, not RSS — %.0f%% of settled RSS is evictable file-backed)\n", (s>0? 100*(s-fs)/s : 0);
    }
  }'
  if [[ "$TIMING" == "1" ]]; then
    echo "  [stage timings]"
    grep '^\[TIMING\]' "$errf" | sed 's/^/    /' || true
  fi
  echo "  spill root: ${KGLITE_TMPDIR:-<system temp>}  (>=256KB columns transit the page cache; see header)"
}

echo "graph: $GRAPH ($(awk -v f="$FILE_BYTES" 'BEGIN{printf "%.1f", f/1048576}') MB)"
echo "csv:   $CSV"
for ((i = 1; i <= REPEAT; i++)); do
  run_once "$i"
done
