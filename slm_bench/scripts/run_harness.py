"""CLI for slm_bench multi-stage harness.

Usage:
    python scripts/run_harness.py \
        --model gemma-4-E2B-it \
        --base-url http://127.0.0.1:8000/v1 \
        --system-prompt configs/system_prompt_default.txt \
        --max-tokens 95 140 230 \
        --margin 30 30 50 \
        --slos 2.0 5.0 10.0 \
        --inputs slm_bench/fixtures/RF390_dataset.jsonl \
        --output results/runs/run_<ts>.jsonl
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import httpx
from tqdm.asyncio import tqdm

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from slm_bench.harness import (  # noqa: E402
    HarnessConfig, load_system_prompt, run_one_prompt, to_jsonl_row,
)


async def main_async(args) -> int:
    sys_prompt = load_system_prompt(args.system_prompt)
    ctk = json.loads(args.chat_template_kwargs) if args.chat_template_kwargs else None

    # Resolve max_tokens / margin: explicit CLI wins; else fall back to preset.
    max_tokens = args.max_tokens
    margin = args.margin
    if args.preset:
        presets_path = Path(args.presets_file)
        if not presets_path.exists():
            raise SystemExit(f"presets file not found: {presets_path}")
        with open(presets_path) as f:
            presets = json.load(f)
        if args.preset not in presets:
            raise SystemExit(
                f"preset {args.preset!r} not in {presets_path} "
                f"(known: {sorted(presets)})"
            )
        ps = presets[args.preset]
        if max_tokens is None:
            max_tokens = ps["max_tokens"]
        if margin is None:
            margin = ps["margin"]

    if max_tokens is None or margin is None:
        raise SystemExit(
            "must supply --max-tokens and --margin, or --preset <name>"
        )

    cfg = HarnessConfig(
        model=args.model,
        base_url=args.base_url,
        system_prompt_text=sys_prompt,
        system_prompt_path=str(Path(args.system_prompt).resolve()),
        max_tokens=max_tokens,
        margin=margin,
        slos=args.slos,
        simulator_backend=args.simulator,
        codex_model=args.codex_model,
        codex_effort=args.codex_effort,
        temperature=args.temperature,
        chat_template_kwargs=ctk,
    )
    print(f"config: {json.dumps(cfg.to_metadata(), indent=2)}", flush=True)

    rows: list[dict] = []
    with open(args.inputs) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if args.limit:
        rows = rows[: args.limit]
    print(f"loaded {len(rows)} prompts from {args.inputs}", flush=True)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"writing trace to {out_path}", flush=True)

    sem = asyncio.Semaphore(args.concurrency)
    timeout = httpx.Timeout(connect=10.0, read=180.0, write=30.0, pool=30.0)
    limits = httpx.Limits(max_connections=args.concurrency * 2)

    async def worker(row: dict) -> str:
        async with sem:
            res = await run_one_prompt(client, cfg, row)
            return to_jsonl_row(res)

    n_done = 0
    with open(out_path, "w") as out_f:
        async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
            tasks = [worker(r) for r in rows]
            for coro in tqdm(
                asyncio.as_completed(tasks), total=len(tasks), desc="harness",
            ):
                line = await coro
                out_f.write(line + "\n")
                out_f.flush()
                n_done += 1

    print(f"\nwrote {n_done} rows to {out_path}", flush=True)
    return 0


def main() -> int:
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", required=True,
                   help="vLLM served-model-name")
    p.add_argument("--base-url", default="http://127.0.0.1:8000/v1",
                   help="OpenAI-compatible endpoint")
    p.add_argument("--system-prompt", required=True,
                   help="Path to system prompt text file")
    p.add_argument("--max-tokens", type=int, nargs=3, default=None,
                   metavar=("S0", "S1", "S2"),
                   help="Per-stage max_tokens (3 values). "
                        "Optional if --preset is given.")
    p.add_argument("--margin", type=int, nargs=3, default=None,
                   metavar=("S0", "S1", "S2"),
                   help="Per-stage margin (3 values). "
                        "Optional if --preset is given.")
    p.add_argument("--preset", default=None,
                   help="Model preset key (in presets file) to load "
                        "max_tokens/margin defaults from. Explicit "
                        "--max-tokens/--margin override.")
    p.add_argument("--presets-file",
                   default=str(ROOT / "configs" / "model_presets.json"),
                   help="Path to presets JSON")
    p.add_argument("--slos", type=float, nargs=3,
                   default=[2.0, 5.0, 10.0],
                   metavar=("S0", "S1", "S2"),
                   help="Cumulative wall-clock SLO seconds")
    p.add_argument("--inputs", required=True,
                   help="Input prompts JSONL (each line: {id, prompt, ...})")
    p.add_argument("--output", default=None,
                   help="Output JSONL path. Default: results/runs/run_<ts>.jsonl")
    p.add_argument("--simulator", default="codex",
                   choices=["codex"],
                   help="ASK simulator backend")
    p.add_argument("--codex-model", default="gpt-5.4")
    p.add_argument("--codex-effort", default="medium",
                   choices=["low", "medium", "high", "xhigh"])
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--chat-template-kwargs", default=None,
                   help="JSON string forwarded to vLLM (e.g. "
                        "'{\"enable_thinking\":false}' for Qwen3)")
    p.add_argument("--concurrency", type=int, default=2,
                   help="Parallel prompts. Note: simulator (codex) is heavy; keep low.")
    p.add_argument("--limit", type=int, default=0,
                   help="0 = all prompts, otherwise process first N")
    args = p.parse_args()

    if args.output is None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        args.output = str(ROOT.parent / "results" / "runs" / f"run_{ts}.jsonl")

    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())
