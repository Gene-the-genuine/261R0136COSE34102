"""Render a slm_bench JSONL run into a single self-contained HTML report.

Usage:
    python scripts/render_html.py results/runs/run_<ts>.jsonl
    python scripts/render_html.py results/runs/run_<ts>.jsonl -o report.html

No external assets — embedded CSS/JS, opens in any browser.
"""
from __future__ import annotations

import argparse
import html
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from slm_bench.harness import reconstruct_messages_in  # noqa: E402

# --------------------------------------------------------------------------
# Styling
# --------------------------------------------------------------------------

ACTION_COLORS = {
    "SOLVE": "#2ea44f",
    "REJECT": "#cf222e",
    "ASK": "#bf8700",
    "THINK": "#218bff",
    None: "#6e7781",
}
END_REASON_COLORS = {
    "solve_action": "#2ea44f",
    "reject_action": "#cf222e",
    "error": "#a40e26",
}

CSS = """
* { box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica,
       Arial, sans-serif; color: #1f2328; background: #f6f8fa; margin: 0;
       padding: 24px; }
h1 { font-size: 22px; margin: 0 0 8px 0; }
h2 { font-size: 16px; margin: 16px 0 8px 0; color: #57606a; }
.summary { background: white; padding: 16px; border-radius: 6px;
           border: 1px solid #d0d7de; margin-bottom: 16px; }
.summary-row { display: flex; flex-wrap: wrap; gap: 12px; font-size: 13px;
               color: #57606a; margin-top: 8px; }
.summary-row .kv { background: #f6f8fa; padding: 4px 10px; border-radius: 4px;
                   border: 1px solid #d0d7de; }
.summary-row .kv b { color: #1f2328; }
.controls { background: white; padding: 12px 16px; border-radius: 6px;
            border: 1px solid #d0d7de; margin-bottom: 16px;
            display: flex; gap: 12px; flex-wrap: wrap; align-items: center; }
.controls input, .controls select { padding: 6px 10px; border-radius: 4px;
                                     border: 1px solid #d0d7de; font: inherit; }
.controls .count { color: #57606a; font-size: 13px; margin-left: auto; }
details.prompt { background: white; border: 1px solid #d0d7de;
                 border-radius: 6px; margin-bottom: 8px; overflow: hidden; }
details.prompt[open] { box-shadow: 0 1px 3px rgba(0,0,0,.06); }
details.prompt > summary { padding: 12px 16px; cursor: pointer; list-style: none;
                            display: flex; gap: 12px; align-items: center;
                            user-select: none; }
details.prompt > summary::-webkit-details-marker { display: none; }
details.prompt > summary::before { content: "▶"; font-size: 10px; color: #6e7781;
                                    transition: transform .15s; }
details.prompt[open] > summary::before { transform: rotate(90deg); }
.pid { font-family: ui-monospace, "SF Mono", Consolas, monospace;
       font-size: 12px; color: #6e7781; min-width: 56px; }
.prompt-text { flex: 1; font-size: 14px;
               white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.badge { display: inline-block; padding: 2px 8px; border-radius: 3px;
         color: white; font-size: 11px; font-weight: 600;
         font-family: ui-monospace, "SF Mono", Consolas, monospace; }
.lat { font-size: 12px; color: #57606a; font-family: ui-monospace, monospace; }
.body { padding: 0 16px 16px 32px; border-top: 1px solid #eaecef; }
.body .label { font-size: 11px; color: #6e7781; text-transform: uppercase;
                letter-spacing: .5px; margin-top: 12px; margin-bottom: 4px; }
.turn { border-left: 3px solid #d0d7de; padding-left: 12px; margin: 12px 0; }
.turn-head { display: flex; gap: 10px; align-items: center; font-size: 12px;
              margin-bottom: 6px; flex-wrap: wrap; }
.phase { background: #f6f8fa; border: 1px solid #d0d7de; border-radius: 4px;
         padding: 8px 12px; margin: 4px 0; font-family: ui-monospace,
         "SF Mono", Consolas, monospace; font-size: 12px;
         white-space: pre-wrap; word-break: break-word; }
.phase.phase2 { background: #fff8e1; border-color: #f0c674; }
.phase.simulator { background: #ddf4ff; border-color: #80ccff; }
.phase.next-user { background: #f0fff4; border-color: #aceebb; }
.phase-meta { font-size: 11px; color: #57606a; margin-bottom: 4px;
               font-family: ui-monospace, monospace; }
.error { color: #cf222e; background: #ffebe9; border: 1px solid #ff8182;
         border-radius: 4px; padding: 8px 12px; margin: 8px 0; font-size: 13px; }
footer { color: #6e7781; font-size: 12px; margin-top: 24px; text-align: center; }
"""

JS = """
function applyFilters() {
  const q = document.getElementById('q').value.toLowerCase();
  const er = document.getElementById('end_reason').value;
  const ac = document.getElementById('action').value;
  const cards = document.querySelectorAll('details.prompt');
  let shown = 0;
  cards.forEach(el => {
    const text = el.dataset.text.toLowerCase();
    const erVal = el.dataset.endReason;
    const actions = el.dataset.actions.split(',');
    const okQ = !q || text.includes(q);
    const okER = !er || erVal === er;
    const okAC = !ac || actions.includes(ac);
    const visible = okQ && okER && okAC;
    el.style.display = visible ? '' : 'none';
    if (visible) shown++;
  });
  document.getElementById('count').textContent = shown + ' / ' + cards.length;
}
document.addEventListener('DOMContentLoaded', () => {
  ['q','end_reason','action'].forEach(id => {
    const el = document.getElementById(id);
    el.addEventListener('input', applyFilters);
    el.addEventListener('change', applyFilters);
  });
  applyFilters();
});
"""


def _h(s) -> str:
    """HTML-escape with None handling."""
    if s is None:
        return ""
    return html.escape(str(s))


def _badge(text: str | None, bg: str) -> str:
    label = text if text is not None else "null"
    return f'<span class="badge" style="background:{bg}">{_h(label)}</span>'


def _phase_block(phase: dict | None, label: str, css_class: str,
                 *, skip_reason: str = "natural EOS in phase 1") -> str:
    if phase is None:
        return f'<div class="phase-meta">{label}: (skipped — {_h(skip_reason)})</div>'
    meta = (
        f"finish_reason=<b>{_h(phase.get('finish_reason'))}</b> "
        f"latency=<b>{phase.get('latency_ms', 0):.0f}ms</b> "
        f"tokens=<b>{phase.get('output_tokens', 0)}</b> "
        f"max={phase.get('request_max_tokens', 0)}"
    )
    return (
        f'<div class="phase-meta">{label}: {meta}</div>'
        f'<div class="phase {css_class}">{_h(phase.get("raw_text", ""))}</div>'
    )


def _simulator_block(sim: dict | None) -> str:
    if not sim:
        return ""
    meta = (
        f"backend=<b>{_h(sim.get('backend'))}</b> "
        f"latency=<b>{sim.get('latency_ms', 0):.0f}ms</b> "
        f"exit={sim.get('exit_code')}"
    )
    return (
        '<div class="label">Simulator (ASK reply)</div>'
        f'<div class="phase-meta">{meta}</div>'
        f'<div class="phase simulator">{_h(sim.get("raw_output", ""))}</div>'
    )


def _turn_block(turn: dict, run: dict, idx: int) -> str:
    state = turn.get("state")
    action = turn.get("action_detected")
    elapsed = turn.get("elapsed_ms", 0)

    head = (
        f'<div class="turn-head">'
        f'<span style="font-weight:600">Turn {idx}</span>'
        f'<span class="pid">state={state}</span>'
        f'{_badge(action, ACTION_COLORS.get(action, ACTION_COLORS[None]))}'
        f'<span class="lat">{elapsed:.0f}ms total</span>'
        f'</div>'
    )

    # full reconstructed messages_in for this turn
    full_msgs = reconstruct_messages_in(run, idx)
    msgs_html = ""
    for m in full_msgs:
        role = m.get("role")
        bg = "#f6f8fa" if role != "user" else "#fff"
        if role == "system":
            bg = "#fff5b1"  # highlight system
        msgs_html += (
            f'<div class="phase-meta">{role}</div>'
            f'<div class="phase" style="background:{bg}">{_h(m.get("content", ""))}</div>'
        )

    body = head
    body += '<details><summary class="label" style="cursor:pointer">'
    body += f'messages_in (reconstructed, {len(full_msgs)} msgs)</summary>'
    body += msgs_html
    body += '</details>'

    body += '<div class="label">Phase 1 (free reasoning)</div>'
    body += _phase_block(turn.get("phase1"), "phase1", "")
    body += '<div class="label">Phase 2 (suffix-injected forced action)</div>'
    p1_fr = (turn.get("phase1") or {}).get("finish_reason")
    if p1_fr == "length":
        skip_reason = "action emitted at length boundary in phase 1 (no SUFFIX needed)"
    else:
        skip_reason = "natural EOS in phase 1"
    body += _phase_block(
        turn.get("phase2"), "phase2", "phase2", skip_reason=skip_reason,
    )

    body += '<div class="label">Full assistant text (phase1+SUFFIX+phase2)</div>'
    body += f'<div class="phase">{_h(turn.get("full_assistant_text", ""))}</div>'

    if turn.get("simulator_call"):
        body += _simulator_block(turn.get("simulator_call"))

    if turn.get("next_user_msg"):
        body += '<div class="label">Next user message (injected for next turn)</div>'
        body += f'<div class="phase next-user">{_h(turn.get("next_user_msg"))}</div>'

    return f'<div class="turn">{body}</div>'


def _prompt_block(run: dict) -> str:
    pid = run.get("id", "?")
    prompt = run.get("prompt", "")
    end_reason = run.get("end_reason", "unknown")
    total = run.get("total_latency_ms", 0)
    n_turns = len(run.get("turns", []))
    actions = [t.get("action_detected") for t in run.get("turns", [])]
    actions_str = ",".join(a or "null" for a in actions)
    er_color = END_REASON_COLORS.get(end_reason)
    if er_color is None:
        er_color = END_REASON_COLORS.get(end_reason.split("_")[0] + "_action", "#6e7781")
        if "terminal" in end_reason:
            er_color = "#9d4edd"

    err_html = ""
    if run.get("error"):
        err_html = f'<div class="error"><b>error:</b> {_h(run["error"])}</div>'

    summary_line = (
        f'<summary>'
        f'<span class="pid">{_h(pid)}</span>'
        f'<span class="prompt-text">{_h(prompt)}</span>'
        f'{_badge(end_reason, er_color)}'
        f'<span class="lat">{total:.0f}ms · {n_turns} turn{"s" if n_turns != 1 else ""}</span>'
        f'</summary>'
    )

    turns_html = "".join(
        _turn_block(t, run, i) for i, t in enumerate(run.get("turns", []))
    )

    safe_text = (prompt or "") + " " + (pid or "") + " " + end_reason
    return (
        f'<details class="prompt" data-text="{_h(safe_text)}" '
        f'data-end-reason="{_h(end_reason)}" data-actions="{_h(actions_str)}">'
        f'{summary_line}<div class="body">{err_html}{turns_html}</div></details>'
    )


def render(jsonl_path: Path, title: str | None = None) -> str:
    runs: list[dict] = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            runs.append(json.loads(line))
    if not runs:
        return "<html><body><p>No data.</p></body></html>"

    cfg = runs[0].get("config", {})
    end_counter = Counter(r.get("end_reason") for r in runs)
    action_counter: Counter = Counter()
    for r in runs:
        for t in r.get("turns", []):
            action_counter[t.get("action_detected") or "null"] += 1
    avg_lat = sum(r.get("total_latency_ms", 0) for r in runs) / max(1, len(runs))

    title = title or f"slm_bench run · {jsonl_path.name}"
    rendered_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    end_reasons = sorted(end_counter.keys())
    actions_seen = sorted(action_counter.keys())

    summary_kvs = "".join(
        f'<span class="kv">{k} <b>{v}</b></span>'
        for k, v in [
            ("model", cfg.get("model")),
            ("system_prompt_sha", cfg.get("system_prompt_sha")),
            ("max_tokens", cfg.get("max_tokens")),
            ("margin", cfg.get("margin")),
            ("slos_cum_s", cfg.get("slos_cumulative_s")),
            ("simulator", cfg.get("simulator_backend")),
        ]
    )

    end_summary = " · ".join(
        f"{er}: <b>{n}</b>" for er, n in sorted(end_counter.items())
    )
    action_summary = " · ".join(
        f"{a}: <b>{n}</b>" for a, n in sorted(action_counter.items())
    )

    end_options = "".join(
        f'<option value="{_h(er)}">{_h(er)} ({end_counter[er]})</option>'
        for er in end_reasons
    )
    action_options = "".join(
        f'<option value="{_h(a)}">{_h(a)} ({action_counter[a]})</option>'
        for a in actions_seen if a != "null"
    )

    cards = "\n".join(_prompt_block(r) for r in runs)

    html_doc = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>{_h(title)}</title>
<style>{CSS}</style></head>
<body>
<h1>{_h(title)}</h1>
<div class="summary">
  <div><b>{len(runs)} prompts</b> · avg latency <b>{avg_lat:.0f}ms</b> · rendered {rendered_at}</div>
  <div class="summary-row">{summary_kvs}</div>
  <div class="summary-row">
    <span class="kv">end_reasons: {end_summary}</span>
    <span class="kv">actions: {action_summary}</span>
  </div>
</div>
<div class="controls">
  <input id="q" placeholder="search prompt / id / end_reason..." style="flex:1; min-width:200px">
  <select id="end_reason"><option value="">all end_reasons</option>{end_options}</select>
  <select id="action"><option value="">any action emitted</option>{action_options}</select>
  <span class="count" id="count">— / {len(runs)}</span>
</div>
{cards}
<footer>slm_bench HTML report · static, no external assets</footer>
<script>{JS}</script>
</body></html>
"""
    return html_doc


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("input", help="JSONL run file")
    p.add_argument("-o", "--output", default=None,
                   help="Output HTML path (default: <input>.html)")
    p.add_argument("--title", default=None)
    args = p.parse_args()

    inp = Path(args.input)
    if not inp.exists():
        print(f"error: {inp} not found", file=sys.stderr)
        return 1
    out = Path(args.output) if args.output else inp.with_suffix(".html")

    out.write_text(render(inp, title=args.title), encoding="utf-8")
    size_kb = out.stat().st_size / 1024
    print(f"wrote {out} ({size_kb:.1f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
