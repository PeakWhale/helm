"""Create the chat model: Ollama (default, local) or any OpenAI-compatible API.

Returns the model plus a short label, or ``None`` and the reason why not, so the
app can fall back to the rule-based planner and tell the user why.
"""

from __future__ import annotations

import json
import logging
import urllib.request
from dataclasses import dataclass
from typing import Any

from .config import LLMSettings

logger = logging.getLogger("helm.llm")

OLLAMA_DEFAULT_URL = "http://localhost:11434"


@dataclass(frozen=True)
class ChatModel:
    model: Any | None
    label: str
    problem: str | None = None


def ollama_problem(base_url: str, model: str, timeout: float = 2.0) -> str | None:
    """Return None when Ollama is up and has the model, else a short reason."""
    url = base_url.rstrip("/") + "/api/tags"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:  # noqa: S310 - local URL from config
            data = json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        return f"Ollama is not reachable at {base_url} ({exc.__class__.__name__}). Start it with `ollama serve`."
    names = {str(m.get("name", "")) for m in data.get("models", [])}
    wanted = model if ":" in model else f"{model}:latest"
    if model in names or wanted in names:
        return None
    return f"Ollama has no model {model!r}. Run: ollama pull {model}"


def build_chat_model(settings: LLMSettings, check: bool = True) -> ChatModel:
    provider = settings.provider
    if provider == "none":
        return ChatModel(None, "none", "LLM is turned off (HELM_LLM_PROVIDER=none).")

    model_name = settings.model_name
    if not model_name:
        return ChatModel(None, provider, "Set HELM_LLM_MODEL to the model name for the OpenAI-compatible API.")

    try:
        if provider == "ollama":
            from langchain_ollama import ChatOllama

            base_url = settings.base_url or OLLAMA_DEFAULT_URL
            if check:
                problem = ollama_problem(base_url, model_name)
                if problem:
                    return ChatModel(None, f"ollama/{model_name}", problem)
            model = ChatOllama(model=model_name, base_url=base_url, temperature=settings.temperature)
            return ChatModel(model, f"ollama/{model_name}")

        from langchain_openai import ChatOpenAI

        api_key = settings.api_key or ("not-needed" if settings.base_url else None)
        if not api_key:
            return ChatModel(None, f"openai/{model_name}", "Set HELM_LLM_API_KEY (or OPENAI_API_KEY).")
        kwargs: dict[str, Any] = {"model": model_name, "api_key": api_key, "temperature": settings.temperature}
        if settings.base_url:
            kwargs["base_url"] = settings.base_url
        return ChatModel(ChatOpenAI(**kwargs), f"openai-compatible/{model_name}")
    except ImportError:
        return ChatModel(None, provider, "LLM support is not installed. Run: uv sync --extra agent")
