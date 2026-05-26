"""Multi-stage SLO action benchmark harness.

State machine (per state_transition.png):
  State 0 ─[Solve|Reject]→ End
  State 0 ─[Ask|Think]→ State 1
  State 1 ─[Solve|Reject]→ End
  State 1 ─[Ask|Think]→ State 2
  State 2 ─[any action]→ End

Each state transition is triggered by EITHER an emitted action token OR
hitting the per-stage token cap. The harness uses two-phase HTTP calls
to the vLLM OpenAI-compatible endpoint (B-arm pattern):
  phase 1: max_tokens = stage_max - stage_margin    (free reasoning)
  phase 2: model emits SUFFIX_TEXT prefilled, max_tokens = stage_margin
           (forces an action header to appear within the stage cap)

The harness records the full message history, both phases of every call,
and the simulator subprocess trace. It does NOT score or judge — the
output JSONL is meant to be parsed and graded by a separate downstream
tool.
"""
from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from slm_bench.parser import SUFFIX_TEXT, extract_action
from slm_bench.simulator import SimulatorCall, simulate_user_response


@dataclass
class HarnessConfig:
    model: str
    base_url: str
    system_prompt_text: str
    system_prompt_path: str | None
    max_tokens: list[int]   # per stage [s0, s1, s2]
    margin: list[int]       # per stage [s0, s1, s2]
    slos: list[float]       # cumulative wall-clock budget [2.0, 5.0, 10.0]
    simulator_backend: str = "codex"
    codex_model: str = "gpt-5.4"
    codex_effort: str = "medium"
    temperature: float = 0.0
    chat_template_kwargs: dict | None = None  # forwarded to vLLM (e.g. {"enable_thinking": False} for Qwen3)

    def __post_init__(self) -> None:
        n = len(self.slos)
        if not (len(self.max_tokens) == len(self.margin) == n):
            raise ValueError(
                f"max_tokens/margin/slos must all have length {n}, got "
                f"{len(self.max_tokens)}/{len(self.margin)}/{n}"
            )
        for k, (mt, mg) in enumerate(zip(self.max_tokens, self.margin)):
            if mg <= 0:
                raise ValueError(f"stage {k}: margin must be > 0, got {mg}")
            if mt <= mg:
                raise ValueError(
                    f"stage {k}: max_tokens({mt}) must be > margin({mg}); "
                    "phase 1 would have 0 token budget"
                )

    @property
    def system_prompt_sha(self) -> str:
        return hashlib.sha256(self.system_prompt_text.encode()).hexdigest()[:16]

    def to_metadata(self) -> dict:
        return {
            "model": self.model,
            "base_url": self.base_url,
            "system_prompt_path": self.system_prompt_path,
            "system_prompt_sha": self.system_prompt_sha,
            "max_tokens": list(self.max_tokens),
            "margin": list(self.margin),
            "slos_cumulative_s": list(self.slos),
            "stage_budgets_s": _stage_budgets(self.slos),
            "simulator_backend": self.simulator_backend,
            "codex_model": self.codex_model,
            "codex_effort": self.codex_effort,
            "temperature": self.temperature,
            "chat_template_kwargs": self.chat_template_kwargs,
        }


@dataclass
class PhaseResult:
    raw_text: str
    finish_reason: str | None
    latency_ms: float
    output_tokens: int
    request_max_tokens: int


@dataclass
class TurnResult:
    state: int
    messages_in: list[dict]
    """Delta of messages newly added at this turn:
       - turn 0: [{role:'system', ...}, {role:'user', content: original prompt}]
       - turn k≥1: [{role:'user', content: simulator reply or '계속.'}]
       Use `reconstruct_messages_in(run, turn_idx)` to rebuild what was
       actually sent to the model at this turn."""
    phase1: PhaseResult | None
    phase2: PhaseResult | None
    full_assistant_text: str
    action_detected: str | None
    next_user_msg: str | None
    simulator_call: SimulatorCall | None
    elapsed_ms: float


@dataclass
class PromptRunResult:
    id: str
    prompt: str
    config: dict
    turns: list[TurnResult]
    end_reason: str
    total_latency_ms: float
    started_at: str
    error: str | None = None


def _stage_budgets(slos_cum: list[float]) -> list[float]:
    out: list[float] = []
    prev = 0.0
    for s in slos_cum:
        out.append(round(s - prev, 6))
        prev = s
    return out


def load_system_prompt(path: str | Path) -> str:
    return Path(path).read_text()


# ---------------------------------------------------------------------------
# vLLM streaming call
# ---------------------------------------------------------------------------


async def _stream_chat(
    client: httpx.AsyncClient,
    base_url: str,
    body: dict,
    timeout_s: float,
) -> tuple[str, str | None, int]:
    """Stream a chat completion. Returns (text, finish_reason, output_tokens).

    Trusts vLLM's `usage` block (sent in the final SSE event when
    `stream_options.include_usage=true`) for the canonical token count.
    """
    body = {**body, "stream": True, "stream_options": {"include_usage": True}}
    text_parts: list[str] = []
    finish_reason: str | None = None
    output_tokens = 0
    base = base_url.rstrip("/")
    async with client.stream(
        "POST", f"{base}/chat/completions", json=body, timeout=timeout_s,
    ) as resp:
        resp.raise_for_status()
        async for line in resp.aiter_lines():
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
            if usage and usage.get("completion_tokens") is not None:
                output_tokens = int(usage["completion_tokens"])
            choices = obj.get("choices") or []
            if not choices:
                continue
            choice = choices[0]
            delta = choice.get("delta") or {}
            content = delta.get("content")
            if content:
                text_parts.append(content)
            if choice.get("finish_reason"):
                finish_reason = choice["finish_reason"]
    if output_tokens == 0:
        output_tokens = max(1, len("".join(text_parts)) // 4)
    return "".join(text_parts), finish_reason, output_tokens


async def _two_phase_call(
    client: httpx.AsyncClient,
    cfg: HarnessConfig,
    messages: list[dict],
    stage: int,
) -> tuple[PhaseResult, PhaseResult | None, str]:
    """One stage's two-phase call. Returns (phase1, phase2_or_None, full_text)."""
    max_tokens = cfg.max_tokens[stage]
    margin = cfg.margin[stage]
    inject_after = max_tokens - margin
    slo = cfg.slos[stage] if stage == 0 else (cfg.slos[stage] - cfg.slos[stage - 1])

    # phase 1
    p1_body = {
        "model": cfg.model,
        "messages": messages,
        "max_tokens": inject_after,
        "temperature": cfg.temperature,
        "seed": 0,
    }
    if cfg.chat_template_kwargs:
        p1_body["chat_template_kwargs"] = cfg.chat_template_kwargs
    t0 = time.perf_counter()
    p1_text, p1_fr, p1_ot = await _stream_chat(
        client, cfg.base_url, p1_body, timeout_s=max(slo * 1.5, 30.0),
    )
    p1_lat = (time.perf_counter() - t0) * 1000
    phase1 = PhaseResult(
        raw_text=p1_text, finish_reason=p1_fr, latency_ms=p1_lat,
        output_tokens=p1_ot, request_max_tokens=inject_after,
    )

    if p1_fr == "stop":
        return phase1, None, p1_text

    # If phase 1 hit the length cap but already emitted an action token,
    # do NOT call phase 2. Injecting SUFFIX_TEXT here would force the model
    # to emit a SECOND action header — extract_action() would still report
    # only the first, while the model's intended final decision in the
    # second slot would be silently masked. Phase 1's raw text is preserved
    # as-is; we just skip an unintended forcing call. (Originally observed
    # in run_20260515T074839Z.html turn 2: phase 1 [SOLVE] then phase 2
    # forced [REJECT], reported as solve_action.)
    if extract_action(p1_text) is not None:
        return phase1, None, p1_text

    # phase 2: prefill SUFFIX_TEXT as start of assistant turn
    cont_messages = list(messages) + [
        {"role": "assistant", "content": p1_text + SUFFIX_TEXT}
    ]
    p2_body = {
        "model": cfg.model,
        "messages": cont_messages,
        "max_tokens": margin,
        "temperature": cfg.temperature,
        "seed": 0,
        "continue_final_message": True,
        "add_generation_prompt": False,
    }
    if cfg.chat_template_kwargs:
        p2_body["chat_template_kwargs"] = cfg.chat_template_kwargs
    t1 = time.perf_counter()
    p2_text, p2_fr, p2_ot = await _stream_chat(
        client, cfg.base_url, p2_body, timeout_s=max(slo * 1.5, 30.0),
    )
    p2_lat = (time.perf_counter() - t1) * 1000
    phase2 = PhaseResult(
        raw_text=p2_text, finish_reason=p2_fr, latency_ms=p2_lat,
        output_tokens=p2_ot, request_max_tokens=margin,
    )
    full = p1_text + SUFFIX_TEXT + p2_text
    return phase1, phase2, full


# ---------------------------------------------------------------------------
# Top-level state machine driver
# ---------------------------------------------------------------------------


async def run_one_prompt(
    client: httpx.AsyncClient,
    cfg: HarnessConfig,
    prompt_row: dict,
) -> PromptRunResult:
    """Drive one prompt through the state machine, end-to-end."""
    pid = str(prompt_row.get("id", ""))
    prompt_text = prompt_row["prompt"]
    started = datetime.now(timezone.utc).isoformat()
    t_start = time.perf_counter()

    messages: list[dict] = [
        {"role": "system", "content": cfg.system_prompt_text},
        {"role": "user", "content": prompt_text},
    ]
    turns: list[TurnResult] = []
    end_reason = "unknown"
    err: str | None = None

    state = 0
    n_states = len(cfg.slos)

    try:
        while state < n_states:
            t_turn = time.perf_counter()
            # delta: only what's newly added this turn
            if state == 0:
                messages_delta = [dict(m) for m in messages]  # [system, user_prompt]
            else:
                messages_delta = [dict(messages[-1])]  # [{role:'user', content:...}]
            phase1, phase2, full_text = await _two_phase_call(
                client, cfg, messages, state,
            )
            action = extract_action(full_text)

            next_user: str | None = None
            sim_call: SimulatorCall | None = None
            advance = False

            is_terminal_state = (state == n_states - 1)

            if action in ("SOLVE", "REJECT"):
                end_reason = f"{action.lower()}_action"
            elif is_terminal_state:
                # state 2: any action terminates (ASK/THINK no further turns)
                end_reason = (
                    f"state{state}_terminal_{action.lower()}" if action
                    else f"state{state}_terminal_no_action"
                )
            elif action == "ASK":
                sim_call = await simulate_user_response(
                    original_prompt=prompt_text,
                    model_ask_text=full_text,
                    backend=cfg.simulator_backend,
                    codex_model=cfg.codex_model,
                    codex_effort=cfg.codex_effort,
                )
                next_user = sim_call.raw_output or "(no response)"
                advance = True
            elif action == "THINK":
                next_user = "계속."
                advance = True
            else:
                # No action parsed and not at terminal — treat like THINK to
                # keep the trace going. Tag the end_reason if this happens
                # at the last state above.
                next_user = "계속."
                advance = True

            turns.append(TurnResult(
                state=state,
                messages_in=messages_delta,
                phase1=phase1,
                phase2=phase2,
                full_assistant_text=full_text,
                action_detected=action,
                next_user_msg=next_user,
                simulator_call=sim_call,
                elapsed_ms=(time.perf_counter() - t_turn) * 1000,
            ))

            if not advance:
                break

            messages.append({"role": "assistant", "content": full_text})
            messages.append({"role": "user", "content": next_user or ""})
            state += 1
    except Exception as exc:  # noqa: BLE001
        err = f"{type(exc).__name__}: {exc}"
        end_reason = "error"

    total_lat = (time.perf_counter() - t_start) * 1000
    return PromptRunResult(
        id=pid,
        prompt=prompt_text,
        config=cfg.to_metadata(),
        turns=turns,
        end_reason=end_reason,
        total_latency_ms=total_lat,
        started_at=started,
        error=err,
    )


# ---------------------------------------------------------------------------
# JSON serialization
# ---------------------------------------------------------------------------


def _json_default(obj):
    if dataclasses.is_dataclass(obj):
        return asdict(obj)
    raise TypeError(f"not serializable: {type(obj).__name__}")


def to_jsonl_row(result: PromptRunResult) -> str:
    return json.dumps(asdict(result), ensure_ascii=False, default=_json_default)


def reconstruct_messages_in(run: dict, turn_idx: int) -> list[dict]:
    """Rebuild the full message list that was sent to the model at `turn_idx`.

    The on-disk JSONL stores `messages_in` as a *delta* per turn:
      - turn 0: [system, user_prompt]
      - turn k≥1: [user (new)]
    To get what the model actually saw at turn k, prepend prior turns:
      [turn0.messages_in] + [turn0.assistant, turn1.messages_in,
       turn1.assistant, turn2.messages_in, ...]
    """
    if not run.get("turns"):
        return []
    msgs: list[dict] = list(run["turns"][0]["messages_in"])
    for j in range(1, turn_idx + 1):
        msgs.append({
            "role": "assistant",
            "content": run["turns"][j - 1]["full_assistant_text"],
        })
        msgs.extend(run["turns"][j]["messages_in"])
    return msgs
