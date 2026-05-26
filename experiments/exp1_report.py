"""Aggregate EXP1 results: 7 cells (RF390 × 3 labels + MMLU × 3 + GSM8K)
into a unified GT% table.

Usage:
    python exp1_report.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score import score_run  # noqa: E402
from score_accuracy import score_accuracy  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results" / "runs"
RF390 = ROOT / "slm_bench" / "fixtures" / "RF390_dataset.jsonl"


def _label_fixture(label: str, out: Path) -> Path:
    with open(RF390) as f, open(out, "w") as g:
        for line in f:
            r = json.loads(line)
            if r["label"] == label:
                g.write(json.dumps(r, ensure_ascii=False) + "\n")
    return out


def main() -> int:
    # Build per-label fixtures for RF390
    rf_fix = {
        "ask": _label_fixture("ask", DATA / "rf390_ask_fixture.jsonl"),
        "reject": _label_fixture("reject", DATA / "rf390_reject_fixture.jsonl"),
        "solve": _label_fixture("solve", DATA / "rf390_solve_fixture.jsonl"),
    }

    cells: list[dict] = []

    # RF390 by label -> score.py (gt count / n_scored)
    for label in ("solve", "ask", "reject"):
        run = DATA / f"run_exp1_rf390_{label}.jsonl"
        if not run.exists():
            cells.append({"cell": f"RF390 - {label.upper()}",
                          "n_scored": 0, "GT%": None,
                          "note": "run missing"})
            continue
        res = score_run(run, rf_fix[label])
        cats = res["by_label"].get(label, {}).get("categories", {})
        gt = cats.get("gt", 0)
        n = res["n_scored"]
        cells.append({
            "cell": f"RF390 - {label.upper()}",
            "n_scored": n,
            "n_total": res["n_fixture"],
            "GT%": (100.0 * gt / n) if n else None,
            "gt": gt, "possible": cats.get("possible", 0),
            "delayed": cats.get("delayed", 0), "wrong": cats.get("wrong", 0),
            "n_error": res["n_error"], "n_missing_run": res["n_missing_run"],
        })

    # MMLU + GSM8K -> score_accuracy
    accuracy_cells = [
        ("MMLU - high_school_biology", "run_exp1_mmlu_bio.jsonl",
         "mmlu_high_school_biology_gt.jsonl"),
        ("MMLU - professional_law", "run_exp1_mmlu_law.jsonl",
         "mmlu_professional_law_gt.jsonl"),
        ("MMLU - abstract_algebra", "run_exp1_mmlu_alg.jsonl",
         "mmlu_abstract_algebra_gt.jsonl"),
        ("GSM8K", "run_exp1_gsm8k.jsonl", "gsm8k_test_gt.jsonl"),
    ]
    for name, run_fn, gt_fn in accuracy_cells:
        run = DATA / run_fn
        gt = DATA / gt_fn
        if not run.exists():
            cells.append({"cell": name, "n_scored": 0, "GT%": None,
                          "note": "run missing"})
            continue
        res = score_accuracy(run, gt)
        cells.append({
            "cell": name,
            "n_scored": res["n_scored"],
            "n_total": res["n_scored"] + res["n_error"] + res["n_missing_gt"],
            "GT%": (100.0 * res["n_correct"] / res["n_scored"])
                   if res["n_scored"] else None,
            "correct": res["n_correct"], "wrong": res["n_wrong"],
            "no_solve": res["n_no_solve"], "n_error": res["n_error"],
        })

    # Print unified table
    print("=" * 80)
    print(f"{'cell':30s} {'n_scored':>8s} {'GT%':>8s}  detail")
    print("-" * 80)
    for c in cells:
        gtpct = f"{c['GT%']:6.2f}%" if c.get("GT%") is not None else "  -   "
        detail = ""
        if "gt" in c:
            detail = (f"gt={c['gt']} poss={c['possible']} "
                      f"delay={c['delayed']} wrong={c['wrong']}")
        elif "correct" in c:
            detail = (f"correct={c['correct']} wrong={c['wrong']} "
                      f"no_solve={c['no_solve']}")
        if c.get("n_error", 0):
            detail += f" err={c['n_error']}"
        if c.get("n_missing_run", 0):
            detail += f" missing={c['n_missing_run']}"
        print(f"{c['cell']:30s} {c.get('n_scored',0):>8d} {gtpct:>8s}  {detail}")
    print("=" * 80)

    # Save JSON for downstream
    with open(DATA / "exp1_report.json", "w") as f:
        json.dump({"cells": cells}, f, ensure_ascii=False, indent=2)
    print(f"\nwrote -> {DATA / 'exp1_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
