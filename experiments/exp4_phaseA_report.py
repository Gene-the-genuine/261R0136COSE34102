"""EXP4 Phase A analysis: 5 models × 2 scales (×1, ×0.5) on RF390.

Output: model × scale × {score, label-wise gt%/poss%/delay%/wrong%,
avg_tok, error_rate, drop(×1→×0.5)}.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score import score_run  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "results" / "runs"
RF390 = Path(__file__).resolve().parents[1] / "slm_bench" / "fixtures" / "RF390_dataset.jsonl"
MODELS = ("gemma", "qwen3-0.6b", "qwen3-1.7b", "qwen3-4b", "phi-3.5-mini")
SCALES = ("1.0", "0.5")


def main() -> int:
    cells: dict[tuple[str, str], dict | None] = {}
    for model in MODELS:
        for scale in SCALES:
            run = DATA / f"run_exp4_{model}_x{scale}.jsonl"
            if not run.exists() or sum(1 for _ in open(run)) < 390:
                cells[(model, scale)] = None
                continue
            res = score_run(run, RF390)
            cells[(model, scale)] = res

    print("=" * 90)
    print(f"=== Score (×1 → ×0.5) ===")
    print(f"  {'model':14s} {'×1':>8s} {'×0.5':>8s}  {'Δ(0.5−1)':>10s}  err_rate")
    for m in MODELS:
        c1 = cells.get((m, "1.0"))
        c5 = cells.get((m, "0.5"))
        s1 = f"{c1['score']:.4f}" if c1 else "    -   "
        s5 = f"{c5['score']:.4f}" if c5 else "    -   "
        if c1 and c5:
            delta = c5["score"] - c1["score"]
            ds = f"{delta:>+10.4f}"
        else:
            ds = "      -   "
        e1 = f"{c1['error_rate']*100:.1f}%" if c1 else "-"
        e5 = f"{c5['error_rate']*100:.1f}%" if c5 else "-"
        print(f"  {m:14s} {s1:>8s} {s5:>8s}  {ds}   ×1={e1}/×0.5={e5}")

    print()
    for lab in ("solve", "ask", "reject"):
        print("=" * 90)
        print(f"=== RF390 {lab.upper()} — gt% (×1 → ×0.5) ===")
        print(f"  {'model':14s} {'×1':>8s} {'×0.5':>8s}  {'Δ':>8s}  ({'×1 cat':<32s} → {'×0.5 cat':<32s})")
        for m in MODELS:
            c1 = cells.get((m, "1.0"))
            c5 = cells.get((m, "0.5"))
            for c in (c1, c5):
                if c is None:
                    continue
            if not c1 or not c5:
                line = f"  {m:14s} (incomplete)"
                print(line)
                continue
            bl1 = c1["by_label"].get(lab, {})
            bl5 = c5["by_label"].get(lab, {})
            n1 = bl1.get("n", 1) or 1
            n5 = bl5.get("n", 1) or 1
            cats1 = bl1.get("categories", {})
            cats5 = bl5.get("categories", {})
            gt1 = 100 * cats1.get("gt", 0) / n1
            gt5 = 100 * cats5.get("gt", 0) / n5
            d = gt5 - gt1
            cs1 = f"gt={cats1.get('gt',0)} p={cats1.get('possible',0)} d={cats1.get('delayed',0)} w={cats1.get('wrong',0)}"
            cs5 = f"gt={cats5.get('gt',0)} p={cats5.get('possible',0)} d={cats5.get('delayed',0)} w={cats5.get('wrong',0)}"
            print(f"  {m:14s} {gt1:>7.1f}% {gt5:>7.1f}%  {d:>+7.1f}%  ({cs1:<32s} → {cs5:<32s})")
    print("=" * 90)

    # Save
    out_path = DATA / "exp4_phaseA_report.json"
    serial = {f"{k[0]}_x{k[1]}": v for k, v in cells.items()}
    with open(out_path, "w") as f:
        json.dump(serial, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nwrote -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
