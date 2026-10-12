"""Optional financial sentiment tool with FinBERT (``uv sync --extra ml``).

The model is loaded on the first call, not at start-up, so the MCP server starts fast.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Callable

logger = logging.getLogger("helm.sentiment")

MODEL_NAME = "ProsusAI/finbert"

# A classifier takes a text and returns {"label": ..., "score": ...}.
Classifier = Callable[[str], dict[str, Any]]


class SentimentError(Exception):
    pass


class SentimentAnalyzer:
    def __init__(self, classifier: Classifier | None = None, model_name: str = MODEL_NAME):
        self._classifier = classifier
        self._model_name = model_name
        self._lock = threading.Lock()

    def _load(self) -> Classifier:
        with self._lock:
            if self._classifier is None:
                try:
                    from transformers import pipeline
                except ImportError as exc:
                    raise SentimentError(
                        "The sentiment tool needs the ml extra: uv sync --extra ml"
                    ) from exc
                logger.info("Loading %s (first call only)...", self._model_name)
                pipe = pipeline("sentiment-analysis", model=self._model_name)
                self._classifier = lambda text: pipe(text[:2000], truncation=True)[0]
            return self._classifier

    def analyze(self, text: str) -> str:
        if not (text or "").strip():
            raise SentimentError("No text to analyze.")
        result = self._load()(text)
        label = str(result.get("label", "unknown")).lower()
        score = float(result.get("score", 0.0))
        return f"Sentiment: {label} (confidence {score:.2f}, model {self._model_name})"
