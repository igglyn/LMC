"""Validated, dependency-free experiment configuration."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TeacherRuntimeConfig:
    """Capacity reserved for LMC's future direct teacher runtime."""

    max_concurrent_requests: int = 1
    request_queue_size: int = 1

    def __post_init__(self) -> None:
        if self.max_concurrent_requests < 1:
            raise ValueError("teacher.max_concurrent_requests must be at least 1")
        if self.request_queue_size < self.max_concurrent_requests:
            raise ValueError("teacher.request_queue_size must cover all workers")


@dataclass(frozen=True)
class ModelRuntimeConfig:
    """Local Hugging Face model artifact and execution settings."""

    id: str
    weights_path: str
    full_weight_training: bool
    device: str = "auto"
    dtype: str = "auto"
    trust_remote_code: bool = False

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("model.id must be non-empty")
        if not self.weights_path:
            raise ValueError("model.weights_path must be non-empty")
        if self.dtype not in {"auto", "float16", "bfloat16", "float32"}:
            raise ValueError("model.dtype must be auto, float16, bfloat16, or float32")


@dataclass(frozen=True)
class ThinkingWindowConfig:
    preserve_kv_for_latent_steps: bool = True
    clear_kv_at_window_end: bool = True
    position_mode: str = "preserve"

    def __post_init__(self) -> None:
        if not self.preserve_kv_for_latent_steps:
            raise ValueError("latent steps must preserve K/V in this experiment")
        if not self.clear_kv_at_window_end:
            raise ValueError("thinking windows must clear K/V at their boundary")
        if self.position_mode not in {"preserve", "compact"}:
            raise ValueError("thinking_window.position_mode must be preserve or compact")


@dataclass(frozen=True)
class ExperimentConfig:
    run_name: str
    seed: int
    model: ModelRuntimeConfig
    teacher: TeacherRuntimeConfig
    thinking_window: ThinkingWindowConfig
    raw: dict[str, Any]

    @classmethod
    def from_path(cls, path: str | Path) -> "ExperimentConfig":
        with Path(path).open(encoding="utf-8") as handle:
            values = json.load(handle)
        return cls.from_dict(values)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "ExperimentConfig":
        for required in ("run_name", "seed", "model", "teacher", "thinking_window"):
            if required not in values:
                raise ValueError(f"missing required configuration key: {required}")
        return cls(
            run_name=str(values["run_name"]),
            seed=int(values["seed"]),
            model=ModelRuntimeConfig(**values["model"]),
            teacher=TeacherRuntimeConfig(**values["teacher"]),
            thinking_window=ThinkingWindowConfig(**values["thinking_window"]),
            raw=values,
        )
