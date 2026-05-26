"""Measure prefill/decode TPS of a running vLLM model via OpenAI API.

Assumes vLLM is already serving on --base-url. The measurement uses a
~300-token prompt (matched to RF390 avg) and a short stream completion,
parsing per-event timestamps to split TTFT (first-token time) from
inter-token delays.

Usage:
    python measure_rates.py --model qwen3-0.6b \
        --base-url http://127.0.0.1:8000/v1 \
        [--out /tmp/rate_qwen3-0.6b.json] [--runs 5]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import httpx

# A ~280-token English+JSON test prompt mirroring RF390 shape.
TEST_PROMPT = (
    "Context:\n"
    "{\n"
    '  "current_signals": {"speed_kph": 42, "limit_kph": 60, '
    '"road": "urban_arterial", "blind_left": "clear", "blind_right": "blocked"},\n'
    '  "policy": "SpeedAssist.setTargetSpeed(target_kph): '
    'target_kph <= active_limit_kph",\n'
    '  "source_priority": ["current_snapshot","policy","route_cache"]\n'
    "}\n\n"
    "User input:\n"
    "Briefly explain whether setting target speed to 55 kph is within the policy. "
    "Reply with one short paragraph (3 sentences max)."
)


def _ts() -> float:
    return time.perf_counter()


def one_run(client: httpx.Client, base_url: str, model: str,
            max_tokens: int = 200,
            extra_body: dict | None = None) -> dict:
    """One streamed completion. Returns timings + token counts."""
    body = {
        "model": model,
        "messages": [{"role": "user", "content": TEST_PROMPT}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if extra_body:
        body.update(extra_body)
    base = base_url.rstrip("/")

    t_send = _ts()
    t_first: float | None = None
    t_last: float | None = None
    text_parts: list[str] = []
    prompt_tokens = 0
    completion_tokens = 0

    with client.stream("POST", f"{base}/chat/completions",
                       json=body, timeout=120.0) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line or not line.startswith("data:"):
                continue
            payload = line[len("data:"):].strip()
            if payload == "[DONE]":
                break
            try:
                obj = json.loads(payload)
            except json.JSONDecodeError:
                continue
            usage = obj.get("usage")
            if usage:
                if usage.get("prompt_tokens") is not None:
                    prompt_tokens = int(usage["prompt_tokens"])
                if usage.get("completion_tokens") is not None:
                    completion_tokens = int(usage["completion_tokens"])
            choices = obj.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            content = delta.get("content")
            if content:
                if t_first is None:
                    t_first = _ts()
                t_last = _ts()
                text_parts.append(content)

    if t_first is None or t_last is None or prompt_tokens == 0 or completion_tokens == 0:
        raise RuntimeError(
            f"incomplete run: prompt_tokens={prompt_tokens} "
            f"completion_tokens={completion_tokens} t_first={t_first}"
        )
    ttft = t_first - t_send
    decode_time = max(t_last - t_first, 1e-6)
    decode_tps = max(completion_tokens - 1, 1) / decode_time
    prefill_tps = prompt_tokens / max(ttft, 1e-6)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "ttft_s": ttft,
        "decode_time_s": decode_time,
        "prefill_tps": prefill_tps,
        "decode_tps": decode_tps,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True,
                   help="vLLM served-model-name (matches --served-model-name)")
    p.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    p.add_argument("--runs", type=int, default=5)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--max-tokens", type=int, default=200)
    p.add_argument("--out", type=Path, default=None,
                   help="Optional JSON output path")
    p.add_argument("--chat-template-kwargs", default=None,
                   help='JSON forwarded to vLLM (e.g. \'{"enable_thinking":false}\')')
    args = p.parse_args()

    extra = json.loads(args.chat_template_kwargs) if args.chat_template_kwargs else None
    extra_body = {"chat_template_kwargs": extra} if extra else None

    with httpx.Client() as client:
        for _ in range(args.warmup):
            _ = one_run(client, args.base_url, args.model,
                        max_tokens=args.max_tokens, extra_body=extra_body)

        runs: list[dict] = []
        for _ in range(args.runs):
            runs.append(one_run(client, args.base_url, args.model,
                                max_tokens=args.max_tokens,
                                extra_body=extra_body))

    def _avg(key: str) -> float:
        return sum(r[key] for r in runs) / len(runs)

    summary = {
        "model": args.model,
        "n_runs": len(runs),
        "prompt_tokens": runs[0]["prompt_tokens"],
        "completion_tokens_avg": _avg("completion_tokens"),
        "ttft_s_avg": _avg("ttft_s"),
        "prefill_tps_avg": _avg("prefill_tps"),
        "decode_tps_avg": _avg("decode_tps"),
        "runs": runs,
    }

    print(json.dumps({k: v for k, v in summary.items() if k != "runs"},
                     indent=2, ensure_ascii=False))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"wrote -> {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
