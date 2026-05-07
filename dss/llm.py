from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Any

from .config import settings

Backend = Literal["openai", "ollama"]


@dataclass
class LlmSpec:
    backend: Backend
    model: str
    temperature: float = settings.temperature


def build_llm(spec: LlmSpec) -> Any:

    if spec.backend == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(model=spec.model, temperature=spec.temperature, format="json")

    raise ValueError(f"Unknown backend: {spec.backend}")