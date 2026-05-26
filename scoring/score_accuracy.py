"""Accuracy scorer for GSM8K/MMLU runs through our 3-step harness.

Only the SOLVE action's payload is considered. Anything else (REJECT, ASK,
delayed/length terminations, errors) -> incorrect (counted as wrong).

Per-task matching:
  gsm8k  : last numeric token in SOLVE payload == ground truth (numeric equality)
  mmlu   : first standalone [A-D] character in SOLVE payload == ground truth letter

Usage:
    python score_accuracy.py <run.jsonl> --gt <gt.jsonl> [--output <out.jsonl>]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


# SOLVE block: text after the [SOLVE] header until next ">>> Action :" or end-of-text.
_SOLVE_BLOCK = re.compile(
    r">>>\s*Action\s*:\s*\[SOLVE\][^\n]*\n?(.*?)(?:>>>\s*Action\s*:|\Z)",
    re.DOTALL,
)
_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
_LETTER = re.compile(r"\b([A-Da-d])\b")

# Priority patterns (most specific first)
_GSM_HASH = re.compile(r"####\s*\$?\s*(-?\d[\d,]*(?:\.\d+)?)")
_GSM_ANSWER = re.compile(
    r"(?:the\s+)?(?:final\s+|correct\s+)?answer\s*"
    r"(?:is|:|=|of)\s*\$?\s*(-?\d[\d,]*(?:\.\d+)?)",
    re.IGNORECASE,
)

_MMLU_ANSWER = re.compile(
    r"(?:the\s+)?(?:final\s+|correct\s+)?answer\s*"
    r"(?:is|:|=)\s*\(?([A-Da-d])\)?",
    re.IGNORECASE,
)
# explicit marker: "(A)", "A)", "A.", "[A]" at start of line/word
_MMLU_MARKER = re.compile(r"(?:^|[\s\n])[\(\[]?([A-Da-d])[\)\]\.]")


def extract_solve_payload(full_assistant_text: str | None) -> str | None:
    if not full_assistant_text:
        return None
    m = _SOLVE_BLOCK.search(full_assistant_text)
    return m.group(1).strip() if m else None


def _norm_num(s: str) -> float | None:
    s = s.replace(",", "").rstrip(".")
    try:
        return float(s)
    except ValueError:
        return None


def gsm8k_extract(payload: str | None) -> tuple[float | None, str]:
    """Return (predicted_number, method). method: hash|answer_kw|last_num|none."""
    if not payload:
        return None, "none"
    m = _GSM_HASH.search(payload)
    if m:
        return _norm_num(m.group(1)), "hash"
    matches = list(_GSM_ANSWER.finditer(payload))
    if matches:
        return _norm_num(matches[-1].group(1)), "answer_kw"
    nums = _NUMBER.findall(payload)
    if nums:
        return _norm_num(nums[-1]), "last_num"
    return None, "none"


def gsm8k_match(payload: str | None, gt: str) -> bool:
    pred, _ = gsm8k_extract(payload)
    truth = _norm_num(gt)
    if pred is None or truth is None:
        return False
    return pred == truth


def mmlu_extract(payload: str | None,
                 choices: list[str] | None = None
                 ) -> tuple[str | None, str]:
    """Return (predicted_letter, method).
    method: answer_kw|marker|choice_text|last_letter|first_letter|none."""
    if not payload:
        return None, "none"
    m = _MMLU_ANSWER.search(payload)
    if m:
        return m.group(1).upper(), "answer_kw"
    m = _MMLU_MARKER.search(payload)
    if m:
        return m.group(1).upper(), "marker"
    if choices:
        # exact substring match of a choice text (must be non-empty and unique)
        hits: list[int] = []
        for i, c in enumerate(choices):
            ct = (c or "").strip()
            if ct and ct in payload:
                hits.append(i)
        if len(hits) == 1:
            return chr(ord("A") + hits[0]), "choice_text"
    letters = _LETTER.findall(payload)
    if letters:
        # prefer last (model often emits reasoning, then final letter)
        return letters[-1].upper(), "last_letter"
    return None, "none"


def mmlu_match(payload: str | None,
               gt_letter: str,
               choices: list[str] | None = None) -> bool:
    pred, _ = mmlu_extract(payload, choices)
    return pred is not None and pred == gt_letter.upper()


def score_accuracy(run_path: Path, gt_path: Path) -> dict:
    gt: dict[str, dict] = {}
    with open(gt_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            gt[row["id"]] = row

    correct = wrong = no_solve = errors = missing_gt = 0
    per_prompt: list[dict] = []

    with open(run_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            pid = r["id"]
            if pid not in gt:
                missing_gt += 1
                continue
            g = gt[pid]
            if r.get("error"):
                errors += 1
                continue

            # find the first SOLVE turn
            payload = None
            for t in r.get("turns") or []:
                if t.get("action_detected") == "SOLVE":
                    payload = extract_solve_payload(t.get("full_assistant_text"))
                    break

            if payload is None:
                no_solve += 1
                ok = False
                gt_val = g.get("answer") or g.get("answer_letter")
                pred = None
                method = "no_solve"
            elif g["task"] == "gsm8k":
                gt_val = g["answer"]
                pred_num, method = gsm8k_extract(payload)
                truth = _norm_num(gt_val)
                ok = (pred_num is not None and truth is not None
                      and pred_num == truth)
                pred = pred_num
            elif g["task"] == "mmlu":
                gt_val = g["answer_letter"]
                pred, method = mmlu_extract(payload, g.get("choices"))
                ok = pred is not None and pred == gt_val.upper()
            else:
                raise ValueError(f"unknown task: {g['task']}")

            if ok:
                correct += 1
            else:
                wrong += 1
            per_prompt.append({
                "id": pid,
                "task": g["task"],
                "ok": ok,
                "method": method,
                "pred": pred,
                "gt": gt_val,
                "payload_head": (payload[:120] if payload else None),
            })

    n_scored = correct + wrong
    return {
        "n_run": correct + wrong + errors,
        "n_scored": n_scored,
        "n_correct": correct,
        "n_wrong": wrong,
        "n_no_solve": no_solve,
        "n_error": errors,
        "n_missing_gt": missing_gt,
        "accuracy": (correct / n_scored) if n_scored else 0.0,
        "per_prompt": per_prompt,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("run_jsonl", type=Path, help="Harness output JSONL")
    p.add_argument("--gt", type=Path, required=True,
                   help="Ground truth JSONL ({id, task, answer/answer_letter})")
    p.add_argument("--output", type=Path, default=None,
                   help="Optional per-prompt JSONL output")
    args = p.parse_args()

    res = score_accuracy(args.run_jsonl, args.gt)
    print("=== accuracy summary ===")
    for k in ("n_run", "n_scored", "n_correct", "n_wrong",
              "n_no_solve", "n_error", "n_missing_gt"):
        print(f"  {k:14s}: {res[k]}")
    print(f"  accuracy      : {res['accuracy']:.4f}")
    print()
    print("=== extraction method × outcome ===")
    from collections import Counter as _C
    mix: _C = _C()
    for r in res["per_prompt"]:
        mix[(r["method"], "ok" if r["ok"] else "ng")] += 1
    width = max(8, max((len(m) for m, _ in mix), default=8))
    print(f"  {'method':<{width}s} {'ok':>5s} {'ng':>5s}")
    methods = sorted({m for m, _ in mix})
    for m in methods:
        ok = mix.get((m, "ok"), 0)
        ng = mix.get((m, "ng"), 0)
        print(f"  {m:<{width}s} {ok:>5d} {ng:>5d}")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            for r in res["per_prompt"]:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"wrote per-prompt -> {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
