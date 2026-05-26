"""EXP2: per-cell distribution of (a) the state where the model emitted
its final SOLVE/REJECT action, and (b) total consumed output tokens.

Linked to EXP1 — operates on the same 7 run_exp1_*.jsonl files.

Definitions:
- final_state: the `state` field of the FIRST turn whose action_detected
  is SOLVE or REJECT (0, 1, or 2). If no such turn exists (ASK/THINK only
  trajectories — i.e. state2_terminal_*), final_state = None.
- final_action: SOLVE | REJECT | None (matches final_state semantics).
- consumed_tokens: sum over all turns of (phase1.output + phase2.output).

Output: console table + JSON dump.
"""
from __future__ import annotations

import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "results" / "runs"

CELLS = [
    ("MMLU - high_school_biology", "run_exp1_mmlu_bio.jsonl"),
    ("MMLU - professional_law", "run_exp1_mmlu_law.jsonl"),
    ("MMLU - abstract_algebra", "run_exp1_mmlu_alg.jsonl"),
    ("GSM8K", "run_exp1_gsm8k.jsonl"),
    ("RF390 - SOLVE", "run_exp1_rf390_solve.jsonl"),
    ("RF390 - ASK", "run_exp1_rf390_ask.jsonl"),
    ("RF390 - REJECT", "run_exp1_rf390_reject.jsonl"),
]


def _percentile(xs: list[float], p: float) -> float | None:
    if not xs:
        return None
    s = sorted(xs)
    k = (len(s) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return float(s[int(k)])
    return s[f] * (c - k) + s[c] * (k - f)


def _avg(xs: list[float]) -> float | None:
    return (sum(xs) / len(xs)) if xs else None


def analyze(run_path: Path) -> dict:
    state_count: Counter = Counter()
    tokens_by_state: dict[object, list[int]] = defaultdict(list)
    tokens_all: list[int] = []
    final_act_dist: Counter = Counter()
    n_scored = 0
    n_error = 0
    with open(run_path) as f:
        for line in f:
            row = json.loads(line)
            if row.get("error"):
                n_error += 1
                continue
            n_scored += 1
            turns = row.get("turns") or []
            final_state = None
            final_act = None
            for t in turns:
                a = t.get("action_detected")
                if a in ("SOLVE", "REJECT"):
                    final_state = t.get("state")
                    final_act = a
                    break
            consumed = 0
            for t in turns:
                p1 = t.get("phase1") or {}
                p2 = t.get("phase2") or {}
                consumed += int(p1.get("output_tokens") or 0)
                consumed += int(p2.get("output_tokens") or 0)
            tokens_all.append(consumed)
            key = final_state if final_state is not None else "none"
            state_count[key] += 1
            tokens_by_state[key].append(consumed)
            final_act_dist[final_act if final_act else "none"] += 1

    # Average solving state — uses only prompts that reached SOLVE/REJECT.
    solved_states = [k for k in state_count if isinstance(k, int)]
    avg_state_solved_only = None
    if solved_states:
        total = 0
        count = 0
        for s in solved_states:
            total += s * state_count[s]
            count += state_count[s]
        if count:
            avg_state_solved_only = total / count

    return {
        "n_scored": n_scored,
        "n_error": n_error,
        "state_count": {str(k): v for k, v in sorted(
            state_count.items(),
            key=lambda x: (isinstance(x[0], str), x[0]))},
        "avg_state_solved_only": avg_state_solved_only,
        "final_action_dist": dict(final_act_dist),
        "tokens_overall": {
            "avg": _avg(tokens_all),
            "p50": _percentile(tokens_all, 50),
            "p90": _percentile(tokens_all, 90),
            "min": min(tokens_all) if tokens_all else None,
            "max": max(tokens_all) if tokens_all else None,
        },
        "tokens_by_state": {
            str(k): {
                "n": len(v),
                "avg": _avg(v),
                "p50": _percentile(v, 50),
                "p90": _percentile(v, 90),
            } for k, v in sorted(
                tokens_by_state.items(),
                key=lambda x: (isinstance(x[0], str), x[0]))
        },
    }


def main() -> int:
    results = {}
    print("=" * 105)
    print(f"{'cell':30s} {'n':>5s} {'s0':>5s} {'s1':>5s} {'s2':>5s} "
          f"{'none':>5s} {'avg_st':>7s}  {'avg_tok':>8s} {'p50':>5s} {'p90':>5s}")
    print("-" * 105)
    for name, fn in CELLS:
        path = DATA / fn
        if not path.exists():
            print(f"{name:30s} (missing)")
            continue
        r = analyze(path)
        results[name] = r
        sc = r["state_count"]
        s0 = sc.get("0", 0); s1 = sc.get("1", 0); s2 = sc.get("2", 0)
        sn = sc.get("none", 0)
        avg_st = (f"{r['avg_state_solved_only']:.2f}"
                  if r["avg_state_solved_only"] is not None else "  -  ")
        tok = r["tokens_overall"]
        print(f"{name:30s} {r['n_scored']:>5d} "
              f"{s0:>5d} {s1:>5d} {s2:>5d} {sn:>5d} "
              f"{avg_st:>7s}  "
              f"{(tok['avg'] or 0):>8.1f} "
              f"{int(tok['p50'] or 0):>5d} {int(tok['p90'] or 0):>5d}")
    print("=" * 105)

    print("\n=== tokens by state (per cell) ===")
    for name in (n for n, _ in CELLS):
        if name not in results:
            continue
        r = results[name]
        print(f"\n[{name}]")
        for st, s in r["tokens_by_state"].items():
            print(f"  state={st:5s} n={s['n']:>4d}  "
                  f"avg={s['avg']:.1f}  p50={int(s['p50'])}  p90={int(s['p90'])}")

    print("\n=== final_action distribution (per cell) ===")
    for name in (n for n, _ in CELLS):
        if name not in results:
            continue
        r = results[name]
        print(f"  {name:30s}  {dict(r['final_action_dist'])}")

    out = DATA / "exp2_report.json"
    with open(out, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nwrote -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
