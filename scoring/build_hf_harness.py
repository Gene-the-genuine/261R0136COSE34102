"""Convert HF datasets (GSM8K + MMLU 3 subjects) into harness-input JSONL
plus separate ground-truth files for accuracy scoring.

Output layout (under results/derived_fixtures/):
    gsm8k_test_prompts.jsonl              {id, prompt}      <- harness input
    gsm8k_test_gt.jsonl                   {id, task, answer}
    mmlu_<subject>_prompts.jsonl          {id, prompt}
    mmlu_<subject>_gt.jsonl               {id, task, answer_letter, choices}
"""
from __future__ import annotations

import json
import re
from pathlib import Path

FIX_HF = Path(__file__).resolve().parents[1] / "slm_bench" / "fixtures" / "hf"
OUT = Path(__file__).resolve().parents[1] / "results" / "derived_fixtures"
OUT.mkdir(parents=True, exist_ok=True)

MMLU_SUBJECTS = ("high_school_biology", "professional_law", "abstract_algebra")
LETTERS = ("A", "B", "C", "D")

_GSM_GT = re.compile(r"####\s*([-\d,\.]+)")


def _gsm_gt(answer_text: str) -> str:
    m = _GSM_GT.search(answer_text)
    if not m:
        raise ValueError(f"no #### marker in: {answer_text!r}")
    return m.group(1).replace(",", "").strip()


def _gsm_prompt(question: str) -> str:
    return f"{question}\n\nReply with just the final number."


def _mmlu_prompt(question: str, choices: list[str]) -> str:
    body = "\n".join(f"{LETTERS[i]}) {c}" for i, c in enumerate(choices))
    return (
        f"Question: {question}\n\n"
        f"Choices:\n{body}\n\n"
        f"Reply with just the letter (A/B/C/D)."
    )


def convert_gsm8k() -> int:
    src = FIX_HF / "gsm8k_test.jsonl"
    out_p = OUT / "gsm8k_test_prompts.jsonl"
    out_g = OUT / "gsm8k_test_gt.jsonl"
    n = 0
    with open(src) as f, open(out_p, "w") as fp, open(out_g, "w") as fg:
        for line in f:
            r = json.loads(line)
            fp.write(json.dumps(
                {"id": r["id"], "prompt": _gsm_prompt(r["question"])},
                ensure_ascii=False,
            ) + "\n")
            fg.write(json.dumps(
                {"id": r["id"], "task": "gsm8k", "answer": _gsm_gt(r["answer"])},
                ensure_ascii=False,
            ) + "\n")
            n += 1
    print(f"GSM8K: {n} -> {out_p.name}, {out_g.name}")
    return n


def convert_mmlu() -> int:
    total = 0
    for sub in MMLU_SUBJECTS:
        src = FIX_HF / f"mmlu_{sub}_test.jsonl"
        out_p = OUT / f"mmlu_{sub}_prompts.jsonl"
        out_g = OUT / f"mmlu_{sub}_gt.jsonl"
        n = 0
        with open(src) as f, open(out_p, "w") as fp, open(out_g, "w") as fg:
            for line in f:
                r = json.loads(line)
                letter = LETTERS[int(r["answer_idx"])]
                fp.write(json.dumps(
                    {"id": r["id"], "prompt": _mmlu_prompt(r["question"], r["choices"])},
                    ensure_ascii=False,
                ) + "\n")
                fg.write(json.dumps(
                    {"id": r["id"], "task": "mmlu", "answer_letter": letter,
                     "choices": r["choices"]},
                    ensure_ascii=False,
                ) + "\n")
                n += 1
        print(f"MMLU/{sub}: {n}")
        total += n
    return total


if __name__ == "__main__":
    g = convert_gsm8k()
    m = convert_mmlu()
    print(f"total: {g + m} rows")
