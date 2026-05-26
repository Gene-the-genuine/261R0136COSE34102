"""User simulator for ASK actions.

When the model emits [ASK], the harness needs a synthetic user reply.
Currently backed by Codex CLI (gpt-5.4). The interface is a single
function `simulate_user_response` returning a structured trace block
that the harness embeds into the run JSONL.
"""
from __future__ import annotations

import asyncio
import shlex
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

SIMULATOR_PROMPT_TEMPLATE = """You are simulating a real end-user who sent the following task to an AI agent:

\"\"\"
{original_prompt}
\"\"\"

The agent responded by asking for clarification:

\"\"\"
{model_ask_text}
\"\"\"

Provide a brief (1-2 sentence) realistic answer to the agent's clarification, in the user's voice. Do not add any meta-commentary, explanations, or quotation marks — just the answer text the user would type back.
"""


@dataclass
class SimulatorCall:
    backend: str
    command: str
    raw_output: str
    latency_ms: float
    out_file: str
    exit_code: int


async def simulate_user_response(
    original_prompt: str,
    model_ask_text: str,
    *,
    backend: str = "codex",
    codex_model: str = "gpt-5.4",
    codex_effort: str = "medium",
    timeout_s: float = 120.0,
    out_dir: Path | str = "/tmp",
) -> SimulatorCall:
    """Invoke the user simulator and return the assistant reply.

    Runs the external CLI in a subprocess (asyncio). Output goes to a
    temp file via codex's `-o` flag and is read back. Returns a
    SimulatorCall with full trace for embedding into the harness JSONL.
    """
    if backend != "codex":
        raise NotImplementedError(f"backend {backend!r} not implemented yet")

    sim_prompt = SIMULATOR_PROMPT_TEMPLATE.format(
        original_prompt=original_prompt,
        model_ask_text=model_ask_text,
    )
    out_path = Path(out_dir) / f"slm_bench_sim_{uuid.uuid4().hex}.md"

    cmd = [
        "codex", "exec",
        "-m", codex_model,
        "-c", f'reasoning.effort="{codex_effort}"',
        "--dangerously-bypass-approvals-and-sandbox",
        "--ephemeral",
        "-o", str(out_path),
        sim_prompt,
    ]
    cmd_redacted = " ".join(shlex.quote(c) for c in cmd[:8]) + " ...<prompt>..."

    t0 = time.perf_counter()
    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    try:
        await asyncio.wait_for(proc.wait(), timeout=timeout_s)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        latency = (time.perf_counter() - t0) * 1000
        return SimulatorCall(
            backend=backend, command=cmd_redacted, raw_output="",
            latency_ms=latency, out_file=str(out_path), exit_code=-1,
        )
    latency = (time.perf_counter() - t0) * 1000

    raw = ""
    if out_path.exists():
        try:
            raw = out_path.read_text().strip()
        except Exception:
            raw = ""

    return SimulatorCall(
        backend=backend, command=cmd_redacted, raw_output=raw,
        latency_ms=latency, out_file=str(out_path), exit_code=proc.returncode or 0,
    )


def to_dict(call: SimulatorCall) -> dict:
    return asdict(call)
