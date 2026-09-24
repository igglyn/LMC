"""Normalization from explicit harness traces to thinking-window pairs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from .schema import RawTrace, ThoughtResponsePair


def trace_to_pair(trace: RawTrace) -> ThoughtResponsePair:
    """Extract one explicitly labeled thinking window and final response.

    The harness labels assistant messages with ``phase: thinking`` and
    ``phase: final``.  Requiring those labels avoids guessing about hidden
    reasoning or tool protocol details owned by the harness.
    """

    thinking_indexes = [i for i, message in enumerate(trace.messages) if message.phase == "thinking"]
    final_indexes = [i for i, message in enumerate(trace.messages) if message.phase == "final"]
    if len(thinking_indexes) != 1 or len(final_indexes) != 1:
        raise ValueError(f"trace {trace.id!r} needs exactly one thinking and one final message")
    thinking_index, final_index = thinking_indexes[0], final_indexes[0]
    if thinking_index >= final_index:
        raise ValueError(f"trace {trace.id!r} has a final response before its thinking window")
    return ThoughtResponsePair(
        id=trace.id,
        context=trace.messages[:thinking_index],
        thinking=trace.messages[thinking_index].content,
        response=trace.messages[final_index].content,
        metadata=trace.metadata,
    )


def read_jsonl(path: str | Path) -> Iterator[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"invalid JSON on line {line_number} of {path}") from error


def normalize_jsonl(input_path: str | Path, output_path: str | Path) -> int:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    seen_ids: set[str] = set()
    with output.open("w", encoding="utf-8") as handle:
        for value in read_jsonl(input_path):
            pair = trace_to_pair(RawTrace.from_dict(value))
            if pair.id in seen_ids:
                raise ValueError(f"duplicate trace id: {pair.id}")
            seen_ids.add(pair.id)
            handle.write(json.dumps(pair.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def validate_pairs(path: str | Path) -> int:
    count = 0
    seen_ids: set[str] = set()
    for value in read_jsonl(path):
        identifier = value.get("id")
        if not isinstance(identifier, str) or not identifier:
            raise ValueError("pair id must be a non-empty string")
        if identifier in seen_ids:
            raise ValueError(f"duplicate pair id: {identifier}")
        if not isinstance(value.get("thinking"), str) or not isinstance(value.get("response"), str):
            raise ValueError(f"pair {identifier!r} needs string thinking and response fields")
        if not isinstance(value.get("context"), list):
            raise ValueError(f"pair {identifier!r} needs a context list")
        seen_ids.add(identifier)
        count += 1
    return count
