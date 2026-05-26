"""Compose per-model max_tokens/margin presets from measured rate JSONs.

Each input is a JSON file produced by scoring/measure_rates.py.
Output is a single presets.json keyed by served-model-name, suitable
for run_harness.py --preset <name>.

Formula:
    per_stage_slo  = diff of cumulative SLOs (e.g. [2.0, 3.0, 5.0])
    stage 0 budget = (S0 - TTFT) * decode_tps * safety_factor
    stage k>0     = Sk * decode_tps * safety_factor   (prefix cache hit assumed)
    margin        = fixed phase2 reserve (30/30/50 by default)

Ensures max_tokens[k] > margin[k] so phase 1 always has positive budget.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

SLOS_CUMUL = (2.0, 5.0, 10.0)
PHASE2_MARGIN = (30, 30, 50)
SAFETY = 0.85


def per_stage_slo(cumul: tuple[float, ...]) -> list[float]:
    out, prev = [], 0.0
    for s in cumul:
        out.append(round(s - prev, 6))
        prev = s
    return out


def compute_preset(
    rate_info: dict,
    slos: tuple[float, ...] = SLOS_CUMUL,
    phase2_margin: tuple[int, ...] = PHASE2_MARGIN,
    safety: float = SAFETY,
) -> dict:
    per_stage = per_stage_slo(slos)
    decode_tps = float(rate_info["decode_tps_avg"])
    ttft = float(rate_info["ttft_s_avg"])
    s0_budget = max(per_stage[0] - ttft, 0.1)
    raw = [
        s0_budget * decode_tps * safety,
        per_stage[1] * decode_tps * safety,
        per_stage[2] * decode_tps * safety,
    ]
    max_tokens = [
        max(int(math.floor(raw[k])), phase2_margin[k] + 5)
        for k in range(3)
    ]
    return {
        "model": rate_info["model"],
        "max_tokens": max_tokens,
        "margin": list(phase2_margin),
        "decode_tps": decode_tps,
        "prefill_tps": float(rate_info["prefill_tps_avg"]),
        "ttft_s": ttft,
        "slos_cumulative_s": list(slos),
        "stage_slos_diff_s": per_stage,
        "safety_factor": safety,
        "prompt_tokens_measured": rate_info["prompt_tokens"],
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("rate_jsons", nargs="+", type=Path,
                   help="Rate JSONs from measure_rates.py")
    p.add_argument("--out", type=Path, required=True,
                   help="Output presets.json (combined)")
    p.add_argument("--safety", type=float, default=SAFETY)
    args = p.parse_args()

    presets: dict[str, dict] = {}
    for path in args.rate_jsons:
        info = json.loads(path.read_text())
        preset = compute_preset(info, safety=args.safety)
        presets[preset["model"]] = preset
        print(
            f"{preset['model']:20s} "
            f"decode={preset['decode_tps']:6.1f} tps  "
            f"ttft={preset['ttft_s']:5.2f}s  "
            f"max_tokens={preset['max_tokens']}  "
            f"margin={preset['margin']}"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(presets, f, ensure_ascii=False, indent=2)
    print(f"\nwrote {len(presets)} presets -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
