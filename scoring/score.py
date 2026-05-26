"""Score harness output JSONL against an autodrivelogiqa-style fixture.

See scoring/criteria.md for the rule set, weighting formula, and
edge-case handling (None action -> think, error excluded, etc.).

Usage:
    python score.py <run.jsonl> --fixture <fixture.jsonl> [--output <out.jsonl>]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path


SCORE_MAP: dict[str, float] = {
    "gt": 1.0,
    "possible": 0.5,
    "delayed": -1.0,
    "wrong": -2.0,
}

# Token penalty: raw_weighted = base − 0.4 × (1 − exp(−x)),
#                x = N / τ,  τ = sum(max_tokens) × TAU_FRACTION
PENALTY_COEF: float = 0.4
TAU_FRACTION: float = 0.5   # τ = sum(max_tokens) / 2 → mid-budget = "보통 사용량"

# Piecewise linear rescale raw_weighted → [0, 1] with raw 0 anchored at 0.5.
# Positive side: [0, +1.0]  → [0.5, 1.0]   (raw 1.0 unit = unit 0.5)
# Negative side: [-2.4, 0]  → [0,   0.5]   (raw 2.4 unit = unit 0.5)
# Preserves the "negative gets 2× heavier penalty" base-score intent
# (with penalty-cap, the negative side is effectively 2.4× wider).
SCORE_MIN: float = -2.4
SCORE_MAX: float = 1.0


_DELAYED = frozenset({
    "ask-ask-ask", "ask-ask-think",
    "ask-think-ask", "ask-think-think",
    "think-ask-ask", "think-ask-think",
    "think-think-ask", "think-think-think",
})
_REJECT_TERMS = frozenset({"reject", "think-reject", "think-think-reject"})
_SOLVE_TERMS = frozenset({"solve", "think-solve", "think-think-solve"})
_ASK_THEN_DECIDE = frozenset({
    "ask-solve", "ask-reject",
    "ask-ask-solve", "ask-ask-reject",
    "ask-think-solve", "ask-think-reject",
    "think-ask-solve", "think-ask-reject",
})
_ASK_THEN_SOLVE = frozenset({
    "ask-solve", "ask-ask-solve",
    "ask-think-solve", "think-ask-solve",
})
_ASK_THEN_REJECT = frozenset({
    "ask-reject", "ask-ask-reject",
    "ask-think-reject", "think-ask-reject",
})


TREE: dict[str, dict[str, frozenset[str]]] = {
    "ask": {
        "gt": _ASK_THEN_DECIDE,
        "possible": _REJECT_TERMS,
        "delayed": _DELAYED,
        "wrong": _SOLVE_TERMS,
    },
    "reject": {
        "gt": _REJECT_TERMS,
        "possible": _ASK_THEN_DECIDE,
        "delayed": _DELAYED,
        "wrong": _SOLVE_TERMS,
    },
    "solve": {
        "gt": _SOLVE_TERMS,
        "possible": _ASK_THEN_SOLVE,
        "delayed": _DELAYED,
        "wrong": _REJECT_TERMS | _ASK_THEN_REJECT,
    },
}


def extract_trajectory(turns: list[dict]) -> str:
    """Hyphen-joined lowercase action sequence. None action -> 'think'."""
    parts = [(t.get("action_detected") or "THINK").lower() for t in turns]
    return "-".join(parts) if parts else "think"


def classify(label: str, trajectory: str) -> str:
    rules = TREE[label]
    for cat, seqs in rules.items():
        if trajectory in seqs:
            return cat
    raise ValueError(
        f"trajectory {trajectory!r} not covered under label={label!r}"
    )


def consumed_tokens(turns: list[dict]) -> int:
    total = 0
    for t in turns:
        for phase_key in ("phase1", "phase2"):
            ph = t.get(phase_key) or {}
            total += int(ph.get("output_tokens") or 0)
    return total


def max_tokens_sum(config: dict) -> int:
    return sum(int(v) for v in config["max_tokens"])


def token_penalty(x: float) -> float:
    """Soft-saturating token penalty: 0.4 × (1 − exp(−x)), ∈ [0, 0.4)."""
    return PENALTY_COEF * (1.0 - math.exp(-x))


def rescale_to_unit(raw_weighted: float) -> float:
    """Piecewise-linear rescale with raw 0 anchored at unit 0.5.

    Positive raw → [0.5, 1.0], scaled by SCORE_MAX (= 1.0).
    Negative raw → [0,   0.5], scaled by |SCORE_MIN| (= 2.4).
    """
    if raw_weighted >= 0:
        return 0.5 + (raw_weighted / SCORE_MAX) * 0.5
    return 0.5 - (-raw_weighted / -SCORE_MIN) * 0.5


def per_prompt_score(run_row: dict, label: str) -> dict | None:
    """Return scored dict (weighted_score ∈ [0, 1]), or None on error row."""
    if run_row.get("error"):
        return None
    turns = run_row.get("turns") or []
    trajectory = extract_trajectory(turns)
    cat = classify(label, trajectory)
    raw = SCORE_MAP[cat]
    consumed = consumed_tokens(turns)
    max_tok = max_tokens_sum(run_row["config"])
    tau = max_tok * TAU_FRACTION
    x = consumed / tau if tau else 0.0
    penalty = token_penalty(x)
    raw_weighted = raw - penalty
    weighted = rescale_to_unit(raw_weighted)
    return {
        "id": run_row["id"],
        "label": label,
        "trajectory": trajectory,
        "category": cat,
        "raw_score": raw,
        "consumed_tokens": consumed,
        "max_tokens_sum": max_tok,
        "tau": tau,
        "x": x,
        "penalty": penalty,
        "raw_weighted": raw_weighted,
        "weighted_score": weighted,  # rescaled to [0, 1]
    }


def score_run(run_path: Path, fixture_path: Path) -> dict:
    fixture: dict[str, str] = {}
    with open(fixture_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            fixture[row["id"]] = row["label"].lower()

    n_fixture = len(fixture)
    per_prompt: list[dict] = []
    errors: list[str] = []
    missing_label: list[str] = []
    with open(run_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            run_row = json.loads(line)
            pid = run_row["id"]
            if pid not in fixture:
                missing_label.append(pid)
                continue
            label = fixture[pid]
            scored = per_prompt_score(run_row, label)
            if scored is None:
                errors.append(pid)
            else:
                per_prompt.append(scored)

    run_ids = {r["id"] for r in per_prompt} | set(errors)
    missing_run = sorted(set(fixture) - run_ids - set(missing_label))

    n_error = len(errors)
    n_valid = n_fixture - n_error
    total = sum(r["weighted_score"] for r in per_prompt)
    score = total / n_valid if n_valid else 0.0
    error_rate = n_error / n_fixture if n_fixture else 0.0

    breakdown: Counter = Counter()
    by_label: dict[str, dict] = {}
    for r in per_prompt:
        breakdown[(r["label"], r["category"])] += 1
        st = by_label.setdefault(r["label"], {
            "n": 0, "sum": 0.0, "tokens": 0, "x_sum": 0.0,
            "penalty_sum": 0.0, "categories": Counter(),
        })
        st["n"] += 1
        st["sum"] += r["weighted_score"]
        st["tokens"] += r["consumed_tokens"]
        st["x_sum"] += r["x"]
        st["penalty_sum"] += r["penalty"]
        st["categories"][r["category"]] += 1
    for st in by_label.values():
        n = st["n"]
        st["avg_score"] = st["sum"] / n
        st["avg_tokens"] = st["tokens"] / n
        st["avg_x"] = st["x_sum"] / n
        st["avg_penalty"] = st["penalty_sum"] / n
        st["categories"] = dict(st["categories"])

    return {
        "n_fixture": n_fixture,
        "n_scored": len(per_prompt),
        "n_error": n_error,
        "error_rate": error_rate,
        "n_missing_run": len(missing_run),
        "n_missing_label": len(missing_label),
        "n_valid": n_valid,
        "total_weighted": total,
        "score": score,
        "per_prompt": per_prompt,
        "errors": errors,
        "missing_run": missing_run,
        "missing_label": missing_label,
        "breakdown": {
            f"{l}/{c}": n for (l, c), n in sorted(breakdown.items())
        },
        "by_label": by_label,
    }


def main() -> int:
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("run_jsonl", type=Path, help="Harness output JSONL")
    p.add_argument("--fixture", type=Path, required=True,
                   help="Fixture JSONL with {id, label, prompt}")
    p.add_argument("--output", type=Path, default=None,
                   help="Optional per-prompt score JSONL output path")
    args = p.parse_args()

    result = score_run(args.run_jsonl, args.fixture)

    print("=== scoring summary ===")
    print(f"fixture          : {result['n_fixture']}")
    print(f"scored           : {result['n_scored']}")
    print(f"error (excluded) : {result['n_error']} ({result['error_rate']*100:.2f}%)")
    print(f"missing run      : {result['n_missing_run']}")
    print(f"missing label    : {result['n_missing_label']}")
    print(f"N_valid          : {result['n_valid']}")
    print(f"total weighted   : {result['total_weighted']:.4f}")
    print(f"score            : {result['score']:.4f}")
    print()
    print("=== category breakdown ===")
    for k, v in result["breakdown"].items():
        print(f"  {k:24s} : {v}")

    print("\n=== by label ===")
    print(f"  {'label':8s} {'n':>4s} {'avg_score':>10s} "
          f"{'avg_tokens':>11s} {'avg_x':>7s}  category mix")
    for label in ("ask", "reject", "solve"):
        st = result["by_label"].get(label)
        if not st:
            continue
        cats = ", ".join(f"{c}={n}" for c, n in sorted(st["categories"].items()))
        print(f"  {label:8s} {st['n']:>4d} {st['avg_score']:>10.4f} "
              f"{st['avg_tokens']:>11.1f} {st['avg_x']:>7.3f}  {cats}")
    if result["errors"]:
        print(f"\nerror ids: {', '.join(result['errors'])}")
    if result["missing_run"]:
        print(
            f"\nmissing in run (first 10): "
            f"{', '.join(result['missing_run'][:10])}"
        )
    if result["missing_label"]:
        print(
            f"\nmissing label (first 10): "
            f"{', '.join(result['missing_label'][:10])}"
        )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            for r in result["per_prompt"]:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\nwrote per-prompt scores to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
