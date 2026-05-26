"""Compare scoring summaries across multiple harness runs.

Usage:
    python compare.py \
        --fixture <fixture.jsonl> \
        --label default --run run_default.jsonl \
        --label cot     --run run_cot.jsonl \
        --label 2x      --run run_2x.jsonl
"""
from __future__ import annotations

import argparse
import sys
from math import nan
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score import score_run  # noqa: E402


def _fmt(v, w=10, prec=4, suffix=""):
    if v != v:  # NaN check
        return f"{'-':>{w}s}"
    return f"{v:>{w}.{prec}f}{suffix}"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--fixture", type=Path, required=True)
    p.add_argument("--label", action="append", required=True,
                   help="One per --run, in paired order")
    p.add_argument("--run", action="append", type=Path, required=True,
                   help="One harness output JSONL per --label")
    args = p.parse_args()

    if len(args.label) != len(args.run):
        print("--label and --run must be paired (same count)", file=sys.stderr)
        return 2

    rows = []
    for label, run_path in zip(args.label, args.run):
        result = score_run(run_path, args.fixture)
        rows.append((label, result))

    LW = max(8, max(len(lab) for lab, _ in rows))

    # overall
    print("=== overall ===")
    print(f"  {'run':<{LW}s} {'N_valid':>7s} {'score':>11s} "
          f"{'total':>9s} {'errors':>7s}")
    for lab, r in rows:
        print(f"  {lab:<{LW}s} {r['n_valid']:>7d} "
              f"{r['score']:>11.4f} {r['total_weighted']:>9.2f} "
              f"{r['n_error']:>7d}")

    # by-label avg_score
    print("\n=== avg_score by label ===")
    print(f"  {'run':<{LW}s} {'ask':>10s} {'reject':>10s} {'solve':>10s}")
    for lab, r in rows:
        bl = r["by_label"]
        a = bl.get("ask", {}).get("avg_score", nan)
        rj = bl.get("reject", {}).get("avg_score", nan)
        sv = bl.get("solve", {}).get("avg_score", nan)
        print(f"  {lab:<{LW}s} {_fmt(a)} {_fmt(rj)} {_fmt(sv)}")

    # by-label avg_tokens
    print("\n=== avg_tokens by label ===")
    print(f"  {'run':<{LW}s} {'ask':>10s} {'reject':>10s} {'solve':>10s}")
    for lab, r in rows:
        bl = r["by_label"]
        a = bl.get("ask", {}).get("avg_tokens", nan)
        rj = bl.get("reject", {}).get("avg_tokens", nan)
        sv = bl.get("solve", {}).get("avg_tokens", nan)
        print(f"  {lab:<{LW}s} {_fmt(a, prec=1)} {_fmt(rj, prec=1)} "
              f"{_fmt(sv, prec=1)}")

    # category mix (overall %)
    print("\n=== category mix (overall %) ===")
    print(f"  {'run':<{LW}s} {'gt':>8s} {'poss':>8s} {'delay':>8s} "
          f"{'wrong':>8s}")
    for lab, r in rows:
        cats = {"gt": 0, "possible": 0, "delayed": 0, "wrong": 0}
        for st in r["by_label"].values():
            for c, n in st["categories"].items():
                cats[c] += n
        total = sum(cats.values()) or 1
        print(
            f"  {lab:<{LW}s} "
            f"{cats['gt']/total*100:>7.1f}% "
            f"{cats['possible']/total*100:>7.1f}% "
            f"{cats['delayed']/total*100:>7.1f}% "
            f"{cats['wrong']/total*100:>7.1f}%"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
