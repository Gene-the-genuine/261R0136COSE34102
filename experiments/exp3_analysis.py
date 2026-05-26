"""EXP3 analysis: 8 system_prompt × RF390, label-별 category 분포 + score.

각 run_exp3_<shot>_<style>.jsonl 을 score.py 로 채점하여,
- label(ASK/SOLVE/REJECT) × category(GT/Possible/Delayed/Wrong) 분포
- avg_score, avg_tokens, gt%, possible%, delayed%, wrong%
를 2×4 (shot × style) 표로 출력.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score import score_run  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "results" / "runs"
RF390 = Path(__file__).resolve().parents[1] / "slm_bench" / "fixtures" / "RF390_dataset.jsonl"

STYLES = ("direct", "cot", "valaware", "default")
SHOTS = ("0shot", "fewshot")
LABELS = ("solve", "ask", "reject")
CATS = ("gt", "possible", "delayed", "wrong")


def main() -> int:
    table = {}  # (shot, style) -> {label -> {cat: count, n, avg_score, avg_tokens}}
    for shot in SHOTS:
        for style in STYLES:
            key = f"{shot}_{style}"
            run = DATA / f"run_exp3_{key}.jsonl"
            if not run.exists():
                table[(shot, style)] = None
                continue
            res = score_run(run, RF390)
            entry = {}
            for label in LABELS:
                st = res["by_label"].get(label, {})
                cats = st.get("categories", {})
                n = st.get("n", 0)
                entry[label] = {
                    "n": n,
                    "avg_score": st.get("avg_score"),
                    "avg_tokens": st.get("avg_tokens"),
                    **{c: cats.get(c, 0) for c in CATS},
                }
            entry["n_scored"] = res["n_scored"]
            entry["n_error"] = res["n_error"]
            entry["score"] = res["score"]
            table[(shot, style)] = entry

    # === Print summary ===
    print("=" * 110)
    print("=== Score (whole RF390) × (shot, style) ===")
    print(f"  {'style':10s} {'0shot':>10s} {'fewshot':>10s}  {'Δ(few−zero)':>12s}")
    for style in STYLES:
        z = table[("0shot", style)]
        f = table[("fewshot", style)]
        zv = z["score"] if z else None
        fv = f["score"] if f else None
        d = (fv - zv) if (zv is not None and fv is not None) else None
        zs = f"{zv:>10.4f}" if zv is not None else f"{'-':>10s}"
        fs = f"{fv:>10.4f}" if fv is not None else f"{'-':>10s}"
        ds = f"{d:>+12.4f}" if d is not None else f"{'-':>12s}"
        print(f"  {style:10s} {zs} {fs} {ds}")

    print()
    for label in LABELS:
        n_label = next((t[label]["n"] for t in table.values() if t), 0)
        print("=" * 110)
        print(f"=== RF390 {label.upper()} (n≈{n_label}) — category distribution % ===")
        print(f"  {'style':10s} {'shot':>9s}  "
              f"{'gt%':>7s} {'poss%':>7s} {'delay%':>7s} {'wrong%':>7s}  "
              f"{'avg_score':>10s}  {'avg_tok':>8s}")
        for style in STYLES:
            for shot in SHOTS:
                t = table.get((shot, style))
                if not t:
                    continue
                st = t[label]
                n = st["n"] or 1
                gt_p = 100 * st["gt"] / n
                ps_p = 100 * st["possible"] / n
                dl_p = 100 * st["delayed"] / n
                wr_p = 100 * st["wrong"] / n
                avs = st["avg_score"] or 0
                avt = st["avg_tokens"] or 0
                print(f"  {style:10s} {shot:>9s}  "
                      f"{gt_p:>6.1f}% {ps_p:>6.1f}% {dl_p:>6.1f}% {wr_p:>6.1f}%  "
                      f"{avs:>+10.4f}  {avt:>8.1f}")
    print("=" * 110)

    out = DATA / "exp3_report.json"
    serial = {f"{k[0]}_{k[1]}": v for k, v in table.items()}
    with open(out, "w") as f:
        json.dump(serial, f, ensure_ascii=False, indent=2)
    print(f"\nwrote -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
