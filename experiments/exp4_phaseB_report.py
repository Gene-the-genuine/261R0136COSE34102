"""EXP4 Phase B analysis: Qwen3-4B × 7 max_tokens scales on RF390.

Output: scale × {score, label-wise gt%/poss%/delay%/wrong%, avg_tok, error_rate}.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score import score_run  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "results" / "runs"
RF390 = Path(__file__).resolve().parents[1] / "slm_bench" / "fixtures" / "RF390_dataset.jsonl"
SCALES = ("0.25", "0.5", "0.75", "1.0", "1.5", "2.0", "3.0")


def main() -> int:
    cells: dict[str, dict] = {}
    for s in SCALES:
        run = DATA / f"run_exp4_qwen3-4b_x{s}.jsonl"
        if not run.exists():
            cells[s] = None
            continue
        cells[s] = score_run(run, RF390)

    print("=" * 100)
    print("=== Score × scale (Qwen3-4B × RF390) ===")
    print(f"  {'scale':>6s}  {'score':>10s}  {'err_rate':>9s}  "
          f"{'avg_tok':>9s}  {'sum_max':>9s}")
    for s in SCALES:
        c = cells[s]
        if c is None:
            print(f"  x{s:>5s}  (missing)")
            continue
        # avg_tokens overall (whole RF390)
        all_tok = sum(st["tokens"] for st in c["by_label"].values())
        all_n = sum(st["n"] for st in c["by_label"].values())
        avg_tok = all_tok / all_n if all_n else 0
        # sum_max from first per_prompt (config)
        first = c["per_prompt"][0] if c["per_prompt"] else None
        sm = first["max_tokens_sum"] if first else 0
        print(f"  x{s:>5s}  {c['score']:>10.4f}  "
              f"{c['error_rate']*100:>8.1f}%  {avg_tok:>9.1f}  {sm:>9d}")

    print()
    for lab in ("solve", "ask", "reject"):
        n_lab = next((c["by_label"][lab]["n"] for c in cells.values() if c), 0)
        print("=" * 100)
        print(f"=== RF390 {lab.upper()} (n={n_lab}) × scale ===")
        print(f"  {'scale':>6s}  {'gt%':>7s} {'poss%':>7s} {'delay%':>7s} {'wrong%':>7s}  "
              f"{'avg_score':>10s} {'avg_tok':>8s}")
        for s in SCALES:
            c = cells[s]
            if c is None:
                print(f"  x{s:>5s}  (missing)")
                continue
            st = c["by_label"][lab]
            n = st["n"] or 1
            cats = st["categories"]
            gt = 100 * cats.get("gt", 0) / n
            ps = 100 * cats.get("possible", 0) / n
            dl = 100 * cats.get("delayed", 0) / n
            wr = 100 * cats.get("wrong", 0) / n
            print(f"  x{s:>5s}  {gt:>6.1f}% {ps:>6.1f}% {dl:>6.1f}% {wr:>6.1f}%  "
                  f"{st['avg_score']:>10.4f} {st['avg_tokens']:>8.1f}")
    print("=" * 100)

    # Save
    out_path = DATA / "exp4_phaseB_report.json"
    serial = {f"x{k}": v for k, v in cells.items()}
    with open(out_path, "w") as f:
        json.dump(serial, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nwrote -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
