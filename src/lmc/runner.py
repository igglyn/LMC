"""Direct local model runner for teacher generation.

This is intentionally a narrow experiment runtime, not an HTTP server.  It
loads one local Hugging Face checkpoint and batches logical teacher requests
against that single model instance.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Protocol, Sequence

from .config import ModelRuntimeConfig, TeacherRuntimeConfig


@dataclass(frozen=True)
class GenerationRequest:
    """One local teacher request, supplied as a JSONL record."""

    id: str
    prompt: str | None = None
    messages: tuple[dict[str, str], ...] = ()
    max_new_tokens: int = 256
    temperature: float = 0.0
    top_p: float = 1.0

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "GenerationRequest":
        identifier = value.get("id")
        prompt = value.get("prompt")
        messages = value.get("messages", [])
        if not isinstance(identifier, str) or not identifier:
            raise ValueError("generation request needs a non-empty id")
        if prompt is not None and not isinstance(prompt, str):
            raise ValueError(f"request {identifier!r} prompt must be a string")
        if not isinstance(messages, list) or any(not isinstance(item, dict) for item in messages):
            raise ValueError(f"request {identifier!r} messages must be a list of objects")
        if bool(prompt) == bool(messages):
            raise ValueError(f"request {identifier!r} needs exactly one of prompt or messages")
        for message in messages:
            if not isinstance(message.get("role"), str) or not isinstance(message.get("content"), str):
                raise ValueError(f"request {identifier!r} messages need string role and content fields")
        request = cls(
            id=identifier,
            prompt=prompt,
            messages=tuple({str(k): str(v) for k, v in item.items()} for item in messages),
            max_new_tokens=int(value.get("max_new_tokens", 256)),
            temperature=float(value.get("temperature", 0.0)),
            top_p=float(value.get("top_p", 1.0)),
        )
        if request.max_new_tokens < 1:
            raise ValueError(f"request {identifier!r} max_new_tokens must be positive")
        if request.temperature < 0:
            raise ValueError(f"request {identifier!r} temperature cannot be negative")
        if not 0 < request.top_p <= 1:
            raise ValueError(f"request {identifier!r} top_p must be in (0, 1]")
        return request


@dataclass(frozen=True)
class GenerationResult:
    id: str
    text: str
    prompt_tokens: int
    completion_tokens: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
        }


class BatchGenerationBackend(Protocol):
    def generate_batch(self, requests: Sequence[GenerationRequest]) -> list[GenerationResult]: ...


class LocalTeacherRunner:
    """Schedules bounded batches against a single loaded model backend."""

    def __init__(self, backend: BatchGenerationBackend, runtime: TeacherRuntimeConfig) -> None:
        self.backend = backend
        self.runtime = runtime

    def generate(self, requests: Sequence[GenerationRequest]) -> list[GenerationResult]:
        ids = [request.id for request in requests]
        if len(ids) != len(set(ids)):
            raise ValueError("generation request ids must be unique")
        results: list[GenerationResult] = []
        batch_size = self.runtime.max_concurrent_requests
        for start in range(0, len(requests), batch_size):
            batch = requests[start : start + batch_size]
            generated = self.backend.generate_batch(batch)
            if [result.id for result in generated] != [request.id for request in batch]:
                raise RuntimeError("model backend must return one ordered result per request")
            results.extend(generated)
        return results


class TransformersGenerationBackend:
    """Local ``transformers`` implementation used by ``LocalTeacherRunner``."""

    def __init__(self, model_config: ModelRuntimeConfig, seed: int) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as error:
            raise RuntimeError(
                "local generation requires the runtime extra; install with `pip install -e '.[runtime]'`"
            ) from error

        self.torch = torch
        self.seed = seed
        self.device = self._resolve_device(model_config.device)
        dtype = self._resolve_dtype(model_config.dtype)
        load_options: dict[str, Any] = {
            "local_files_only": True,
            "trust_remote_code": model_config.trust_remote_code,
        }
        self.tokenizer = AutoTokenizer.from_pretrained(model_config.weights_path, **load_options)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_config.weights_path,
            torch_dtype=dtype,
            **load_options,
        ).to(self.device)
        self.model.eval()

    def _resolve_device(self, configured: str) -> str:
        if configured == "auto":
            return "cuda" if self.torch.cuda.is_available() else "cpu"
        if configured.startswith("cuda") and not self.torch.cuda.is_available():
            raise RuntimeError("model.device requests CUDA but CUDA is unavailable")
        return configured

    def _resolve_dtype(self, configured: str):
        if configured == "auto":
            return "auto"
        return getattr(self.torch, configured)

    def _render_prompt(self, request: GenerationRequest) -> str:
        if request.prompt is not None:
            return request.prompt
        if getattr(self.tokenizer, "chat_template", None):
            return self.tokenizer.apply_chat_template(
                list(request.messages), tokenize=False, add_generation_prompt=True
            )
        return "\n".join(f"{message['role']}: {message['content']}" for message in request.messages) + "\nassistant:"

    def generate_batch(self, requests: Sequence[GenerationRequest]) -> list[GenerationResult]:
        if not requests:
            return []
        settings = {(r.max_new_tokens, r.temperature, r.top_p) for r in requests}
        if len(settings) != 1:
            raise ValueError("each scheduled batch must use identical generation settings")
        max_new_tokens, temperature, top_p = settings.pop()
        encoded = self.tokenizer(
            [self._render_prompt(request) for request in requests],
            return_tensors="pt",
            padding=True,
        ).to(self.device)
        do_sample = temperature > 0
        generation_options: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if do_sample:
            generation_options.update(
                temperature=temperature,
                top_p=top_p,
                generator=self.torch.Generator(device=self.device).manual_seed(self.seed),
            )
        with self.torch.inference_mode():
            output = self.model.generate(**encoded, **generation_options)
        input_width = encoded["input_ids"].shape[1]
        completions = output[:, input_width:]
        texts = self.tokenizer.batch_decode(completions, skip_special_tokens=True)
        attention = encoded["attention_mask"].sum(dim=1).tolist()
        return [
            GenerationResult(
                id=request.id,
                text=text,
                prompt_tokens=int(prompt_tokens),
                completion_tokens=int(completion.shape[0]),
            )
            for request, text, prompt_tokens, completion in zip(requests, texts, attention, completions, strict=True)
        ]


def read_generation_requests(path: str) -> list[GenerationRequest]:
    requests: list[GenerationRequest] = []
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    requests.append(GenerationRequest.from_dict(json.loads(line)))
                except json.JSONDecodeError as error:
                    raise ValueError(f"invalid JSON on line {line_number} of {path}") from error
    return requests
