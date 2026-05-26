"""EXP5: 6 models × RF390, score + label-wise category mix.

Sources:
- Qwen3-4B / Gemma / Qwen3-0.6B / Qwen3-1.7B / Phi-3.5-mini: reused from EXP4 Phase A
- Llama-3.2-3B: run_exp5_llama-3.2-3b.jsonl (measured in this experiment)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score import score_run  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "results" / "runs"
RF390 = Path(__file__).resolve().parents[1] / "slm_bench" / "fixtures" / "RF390_dataset.jsonl"

MODELS = [
    ("gemma",         "run_exp4_gemma_x1.0.jsonl"),
    ("qwen3-0.6b",    "run_exp4_qwen3-0.6b_x1.0.jsonl"),
    ("qwen3-1.7b",    "run_exp4_qwen3-1.7b_x1.0.jsonl"),
    ("qwen3-4b",      "run_exp4_qwen3-4b_x1.0.jsonl"),
    ("llama-3.2-3b",  "run_exp5_llama-3.2-3b.jsonl"),
    ("phi-3.5-mini",  "run_exp4_phi-3.5-mini_x1.0.jsonl"),
]


def main() -> int:
    results: dict[str, dict | None] = {}
    for name, fn in MODELS:
        path = DATA / fn
        if not path.exists() or sum(1 for _ in open(path)) < 390:
            results[name] = None
            print(f"WARN: {name} missing or incomplete ({path})")
            continue
        results[name] = score_run(path, RF390)

    print()
    print("=" * 110)
    print("=== EXP5 — 6 models × RF390 × preset × default SP ===")
    print(f"  {'model':14s} {'score':>7s} {'err':>5s} {'avg_tok':>8s}  "
          f"SOLVE(gt/p/d/w%)            ASK(gt/p/d/w%)              REJECT(gt/p/d/w%)")
    for name, _ in MODELS:
        r = results[name]
        if r is None:
            print(f"  {name:14s}  (missing)")
            continue
        all_tok = sum(st["tokens"] for st in r["by_label"].values())
        all_n = sum(st["n"] for st in r["by_label"].values())
        avg_tok = all_tok / all_n if all_n else 0
        triples = []
        for lab in ("solve", "ask", "reject"):
            st = r["by_label"].get(lab, {"n": 1, "categories": {}})
            n = st["n"] or 1
            c = st["categories"]
            triples.append(
                f"{100*c.get('gt',0)/n:5.1f}/{100*c.get('possible',0)/n:5.1f}/"
                f"{100*c.get('delayed',0)/n:5.1f}/{100*c.get('wrong',0)/n:5.1f}"
            )
        print(f"  {name:14s} {r['score']:>7.4f} "
              f"{r['error_rate']*100:>4.1f}% {avg_tok:>8.1f}  "
              f"{triples[0]:<28s} {triples[1]:<28s} {triples[2]:<28s}")
    print("=" * 110)

    out_path = DATA / "exp5_report.json"
    serial = {name: r for name, r in results.items()}
    with open(out_path, "w") as f:
        json.dump(serial, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nwrote -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
