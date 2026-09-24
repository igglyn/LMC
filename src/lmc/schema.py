"""Trace and derived-pair schemas owned by the training boundary."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class TraceMessage:
    role: str
    content: str
    phase: str | None = None

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "TraceMessage":
        role = str(value.get("role", ""))
        content = value.get("content")
        if not role or not isinstance(content, str):
            raise ValueError("each trace message needs string role and content")
        phase = value.get("phase")
        if phase is not None and phase not in {"thinking", "final"}:
            raise ValueError("message phase must be thinking or final")
        return cls(role=role, content=content, phase=phase)


@dataclass(frozen=True)
class RawTrace:
    id: str
    messages: tuple[TraceMessage, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "RawTrace":
        identifier = value.get("id")
        messages = value.get("messages")
        if not isinstance(identifier, str) or not identifier:
            raise ValueError("trace id must be a non-empty string")
        if not isinstance(messages, list):
            raise ValueError("trace messages must be a list")
        return cls(
            id=identifier,
            messages=tuple(TraceMessage.from_dict(item) for item in messages),
            metadata=dict(value.get("metadata", {})),
        )


@dataclass(frozen=True)
class ThoughtResponsePair:
    id: str
    context: tuple[TraceMessage, ...]
    thinking: str
    response: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["context"] = [asdict(message) for message in self.context]
        return value
