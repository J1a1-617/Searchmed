from __future__ import annotations

import os
from typing import Callable, Dict, List, Sequence, Tuple, TypeVar


T = TypeVar("T")

SKILL_ID = "structured-output-continuation"


def configured_batch_size(env_name: str, default: int) -> int:
    """Return a positive per-call item cap for an oversized structured stage."""
    try:
        return max(1, int(os.environ.get(env_name) or default))
    except (TypeError, ValueError):
        return max(1, default)


def run_partitioned(
    items: Sequence[T],
    *,
    batch_size: int,
    call_part: Callable[[List[T], int, int], Dict],
) -> List[Dict]:
    """Complete one logical structured task through independently valid parts.

    Parts are deliberately fixed before calling the model.  This avoids trying to
    repair an already truncated JSON fragment and keeps every API response a
    complete schema-valid object that can be merged deterministically.
    """
    rows = list(items)
    if not rows:
        return []
    parts = [rows[index:index + batch_size] for index in range(0, len(rows), batch_size)]
    total = len(parts)
    return [call_part(part, number, total) for number, part in enumerate(parts, start=1)]


def is_recoverable_structured_output_error(exc: Exception) -> bool:
    """Return true only for incomplete/invalid structured model output."""
    message = str(exc).lower()
    return any(marker in message for marker in (
        "tool arguments are not valid json",
        "expected exactly one",
        "failed semantic validation",
        "must be json object",
        "claim ids mismatch",
        "missing required structured fields",
        "unterminated string",
        "jsondecodeerror",
    ))


def run_adaptive(
    items: Sequence[T],
    *,
    batch_size: int,
    call_part: Callable[[List[T], int, int], Dict],
) -> Tuple[List[Dict], bool]:
    """Try one complete call, partitioning only after structured-output failure."""
    rows = list(items)
    if not rows:
        return [], False
    try:
        return [call_part(rows, 1, 1)], False
    except Exception as exc:
        if len(rows) <= batch_size or not is_recoverable_structured_output_error(exc):
            raise
        return run_partitioned(rows, batch_size=batch_size, call_part=call_part), True


def unique_strings(values: Sequence[object]) -> List[str]:
    result: List[str] = []
    seen = set()
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def recovery_trace(
    *,
    input_items: int,
    part_count: int,
    batch_size: int,
    initial_attempt_failed: bool = False,
) -> Dict[str, object]:
    return {
        "skill_id": SKILL_ID,
        "activated": initial_attempt_failed,
        "input_items": input_items,
        "part_count": part_count,
        "batch_size": batch_size,
        "initial_attempt_failed": initial_attempt_failed,
        "merge_mode": "deterministic",
    }
