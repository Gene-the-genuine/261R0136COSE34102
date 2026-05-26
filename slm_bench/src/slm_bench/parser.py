"""Action token extraction from raw model output.

This is the only "interpretation" the harness does — pure structural
detection of `>>> Action : [SOLVE|REJECT|ASK|THINK]` to drive the state
machine. No semantic scoring or correctness judgment.
"""
from __future__ import annotations

import re

ACTIONS: tuple[str, ...] = ("SOLVE", "REJECT", "ASK", "THINK")
SUFFIX_TEXT: str = "\n >>> Action :"

ACTION_PATTERN = re.compile(
    r">>>\s*Action\s*:\s*\[(" + "|".join(ACTIONS) + r")\]"
)


def extract_action(text: str | None) -> str | None:
    """Return the first action label found in `text`, or None.

    Only the first match counts — even if the model emits multiple action
    headers (e.g. due to suffix injection collisions), the harness treats
    the first one as the binding decision.
    """
    if not text:
        return None
    m = ACTION_PATTERN.search(text)
    return m.group(1) if m else None
