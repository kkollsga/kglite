"""Compare two Phase 11 harness runs and emit a markdown delta table.

Joins by (name, scale, mode). p50 is the primary comparison; stddev +
p95 shown for context. Flags per-mode gates (memory +5, mapped +10,
disk +15 %) per the Phase 11 plan.

Usage:
    python tests/benchmarks/phase11_compare.py \
        dev_workfolder/dev-documentation/archive/phase_artifacts/phase11_v0_7_17.json \
        dev_workfolder/dev-documentation/archive/phase_artifacts/phase11_main.json \
        > /tmp/phase11_delta.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

GATE_MEMORY = 0.05  # +5 % block
GATE_MAPPED = 0.10  # +10 % block
GATE_DISK = 0.15  # +15 % flag (no block — disk is a new 0.8.0 product)

# Flag-only thresholds (inform, don't block)
FLAG_MEMORY = 0.02
FLAG_MAPPED = 0.05


def _key(cell):
    return (cell.get("name"), cell.get("scale"), cell.get("mode"))


def _fmt_ms(v):
    if v is None:
        return "–"
    return f"{v * 1000:.3f}"


def _fmt_pct(v):
    if v is None:
        return "–"
    sign = "+" if v > 0 else ""
    return f"{sign}{v * 100:.1f}%"


def _gate(mode, delta):
    if delta is None:
        return ""
    thresholds = {
        "memory": (GATE_MEMORY, FLAG_MEMORY),
        "mapped": (GATE_MAPPED, FLAG_MAPPED),
        "disk": (GATE_DISK, None),
    }
    if mode not in thresholds:
        return ""
    block, flag = thresholds[mode]
    if delta >= block:
        return "**block**" if mode != "disk" else "flag"
    if flag is not None and delta >= flag:
        return "flag"
    return "ok"


def load(path: Path):
    with open(path) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("before", type=Path, help="v0.7.17 JSON")
    parser.add_argument("after", type=Path, help="0.8.0 JSON")
    parser.add_argument("--out", type=Path, default=None, help="Write markdown to this file (default: stdout)")
    args = parser.parse_args()

    before = {_key(c): c for c in load(args.before)["cells"]}
    after = {_key(c): c for c in load(args.after)["cells"]}

    before_meta = load(args.before)
    after_meta = load(args.after)

    lines: list[str] = []
    lines.append("### Phase 11 N=20 benchmark delta — v0.7.17 → 0.8.0")
    lines.append("")
    lines.append(f"- Before: `{before_meta.get('git_sha', '?')}` (v{before_meta.get('kglite_version', '?')})")
    lines.append(f"- After: `{after_meta.get('git_sha', '?')}` (v{after_meta.get('kglite_version', '?')})")
    lines.append(f"- N={after_meta.get('n_trials', '?')} trials per cell. All times in milliseconds.")
    lines.append("")

    def _rows_for(predicate, modes=("memory", "mapped", "disk")):
        rows = []
        all_keys = sorted(set(before) | set(after))
        for k in all_keys:
            name, scale, mode = k
            if mode not in modes:
                continue
            if name == "footprint":
                continue
            if not predicate(k):
                continue
            b = before.get(k)
            a = after.get(k)
            b_p50 = b["p50"] if b and "p50" in b else None
            a_p50 = a["p50"] if a and "p50" in a else None
            delta = (a_p50 - b_p50) / b_p50 if (a_p50 and b_p50) else None
            gate = _gate(mode, delta) if delta is not None else ""
            rows.append((name, scale, mode, b_p50, a_p50, delta, gate, a, b))
        return rows

    # Construction section
    lines.append("### Construction sweep (wall-clock build)")
    lines.append("")
    lines.append("| test | scale | mode | v0.7.17 p50 | 0.8.0 p50 | Δ p50 | 0.8.0 p95 | 0.8.0 σ | gate |")
    lines.append("|---|---:|---|---:|---:|---:|---:|---:|---|")
    for name, scale, mode, bp50, ap50, delta, gate, a, b in _rows_for(lambda k: k[0] == "construction"):
        lines.append(
            f"| {name} | {scale} | {mode} | {_fmt_ms(bp50)} | {_fmt_ms(ap50)} | "
            f"{_fmt_pct(delta)} | {_fmt_ms(a['p95']) if a else '–'} | "
            f"{_fmt_ms(a['stddev']) if a else '–'} | {gate} |"
        )
    lines.append("")

    # Query primitives section
    lines.append("### Query primitives at 10k nodes")
    lines.append("")
    lines.append("| test | mode | v0.7.17 p50 | 0.8.0 p50 | Δ p50 | 0.8.0 p95 | 0.8.0 σ | gate |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---|")
    for name, scale, mode, bp50, ap50, delta, gate, a, b in _rows_for(
        lambda k: k[0] not in ("construction", "footprint") and k[1] == 10_000
    ):
        lines.append(
            f"| {name} | {mode} | {_fmt_ms(bp50)} | {_fmt_ms(ap50)} | "
            f"{_fmt_pct(delta)} | {_fmt_ms(a['p95']) if a else '–'} | "
            f"{_fmt_ms(a['stddev']) if a else '–'} | {gate} |"
        )
    lines.append("")

    # Summary
    flagged = [
        (k, ((after[k]["p50"] - before[k]["p50"]) / before[k]["p50"]))
        for k in before
        if k in after
        and k[0] not in ("construction", "footprint")
        and "p50" in before[k]
        and "p50" in after[k]
        and before[k]["p50"] > 0
        and ((after[k]["p50"] - before[k]["p50"]) / before[k]["p50"]) > FLAG_MEMORY
    ]
    wins = [
        (k, ((after[k]["p50"] - before[k]["p50"]) / before[k]["p50"]))
        for k in before
        if k in after
        and k[0] not in ("construction", "footprint")
        and "p50" in before[k]
        and "p50" in after[k]
        and before[k]["p50"] > 0
        and ((after[k]["p50"] - before[k]["p50"]) / before[k]["p50"]) < -FLAG_MEMORY
    ]

    lines.append("### Summary")
    lines.append("")
    lines.append(f"- **Wins** ({len(wins)}): queries at least 2 % faster.")
    for (name, scale, mode), d in sorted(wins, key=lambda x: x[1])[:10]:
        lines.append(f"  - `{name}_{scale}_{mode}`: {_fmt_pct(d)}")
    lines.append(f"- **Flags** ({len(flagged)}): queries at least 2 % slower.")
    for (name, scale, mode), d in sorted(flagged, key=lambda x: -x[1])[:10]:
        lines.append(f"  - `{name}_{scale}_{mode}`: {_fmt_pct(d)}")

    out = "\n".join(lines) + "\n"
    if args.out:
        args.out.write_text(out)
        print(f"wrote {args.out}", file=sys.stderr)
    else:
        sys.stdout.write(out)


if __name__ == "__main__":
    main()
