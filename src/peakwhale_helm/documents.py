"""Search over a folder of documents (PDF, Markdown, TXT).

Files are split into chunks. Each chunk keeps its source file, its chunk number
and (for PDFs) its page, so every search hit can be cited.

Two retrievers:

* ``bm25`` (default) - keyword search in pure Python. No model, no download.
* ``embeddings`` - semantic search with a sentence-transformers model
  (needs ``uv sync --extra ml``). Any object with ``embed_documents`` and
  ``embed_query`` methods can be passed in instead, which is how the tests
  use a small fake model.

The index is built on the first search and rebuilt when a file changes.
"""

from __future__ import annotations

import logging
import math
import re
import threading
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

logger = logging.getLogger("helm.documents")

SUPPORTED_SUFFIXES = (".pdf", ".md", ".markdown", ".txt")

STOPWORDS = frozenset(
    """a an and are as at be by can could did do does for from had has have how i if in into is it
    its me my of on or our over please show so tell than that the their them then there these they
    this to us was we were what when where which who whom why will with would you your about any
    find get give look list""".split()
)


class DocumentError(Exception):
    """A problem the user can fix. The message is shown as is."""


@dataclass(frozen=True)
class Chunk:
    source: str  # path relative to the documents folder, with "/" separators
    index: int  # chunk number inside the file, starting at 0
    text: str
    page: int | None = None  # 1-based page number for PDFs

    @property
    def label(self) -> str:
        page = f" (page {self.page})" if self.page else ""
        return f"{self.source}, chunk {self.index}{page}"


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float


@dataclass(frozen=True)
class DocumentInfo:
    source: str
    kind: str
    size_bytes: int
    chunks: int
    error: str | None = None


class Embedder(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


# ---------------------------------------------------------------------- text helpers


def _stem(word: str) -> str:
    """A very small English stemmer. Query and text use the same rules, so it only
    needs to be consistent, not correct ("declares", "declared" -> "declar")."""
    if len(word) > 4 and word.endswith("ies"):
        word = word[:-3] + "y"
    elif len(word) > 4 and word.endswith("es") and word[:-2].endswith(("s", "x", "z", "ch", "sh")):
        word = word[:-2]
    elif len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "us", "is")):
        word = word[:-1]
    if len(word) > 5 and word.endswith("ing"):
        word = word[:-3]
    elif len(word) > 4 and word.endswith("ed"):
        word = word[:-2]
    if len(word) > 4 and word.endswith("e"):
        word = word[:-1]
    return word


def terms(text: str, drop_stopwords: bool = True) -> list[str]:
    words = re.findall(r"[0-9a-zA-Z]+", (text or "").lower())
    return [_stem(w) for w in words if not (drop_stopwords and w in STOPWORDS)]


def chunk_text(text: str, size: int = 800, overlap: int = 120) -> list[str]:
    """Split text into chunks of about ``size`` characters.

    Paragraphs are kept together when they fit. Long paragraphs are split on
    sentence ends, then on spaces. Neighbouring chunks share ``overlap``
    characters so a fact on the border is not lost.
    """
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    pieces: list[str] = []
    for paragraph in paragraphs:
        if len(paragraph) <= size:
            pieces.append(paragraph)
            continue
        sentences = re.split(r"(?<=[.!?])\s+", paragraph)
        for sentence in sentences:
            while len(sentence) > size:
                cut = sentence.rfind(" ", 0, size)
                cut = cut if cut > size // 2 else size
                pieces.append(sentence[:cut].strip())
                sentence = sentence[cut:].strip()
            if sentence:
                pieces.append(sentence)

    chunks: list[str] = []
    current = ""
    for piece in pieces:
        joiner = "\n\n" if current else ""
        if len(current) + len(joiner) + len(piece) <= size:
            current = current + joiner + piece
            continue
        if current:
            chunks.append(current)
            tail = current[-overlap:] if overlap else ""
            if tail and " " in tail:
                tail = tail[tail.find(" ") + 1 :]
            current = (tail + "\n\n" + piece).strip() if tail and len(tail) + len(piece) + 2 <= size else piece
        else:
            current = piece
    if current:
        chunks.append(current)
    return chunks


def read_document(path: Path) -> list[tuple[int | None, str]]:
    """Return (page, text) pairs. Page is None for text files."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        return [(i + 1, page.extract_text() or "") for i, page in enumerate(reader.pages)]
    return [(None, path.read_text(encoding="utf-8", errors="replace"))]


# ---------------------------------------------------------------------- retrievers


class BM25Index:
    def __init__(self, chunks: Sequence[Chunk], k1: float = 1.5, b: float = 0.75):
        self.chunks = list(chunks)
        self.k1, self.b = k1, b
        self.doc_terms = [Counter(terms(c.text + " " + c.source)) for c in self.chunks]
        self.lengths = [sum(t.values()) for t in self.doc_terms]
        self.avg_len = (sum(self.lengths) / len(self.lengths)) if self.lengths else 0.0
        df: Counter[str] = Counter()
        for counts in self.doc_terms:
            df.update(counts.keys())
        n = len(self.chunks)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: str) -> list[float]:
        query_terms = terms(query)
        result = []
        for counts, length in zip(self.doc_terms, self.lengths):
            score = 0.0
            for term in query_terms:
                tf = counts.get(term, 0)
                if not tf:
                    continue
                norm = tf + self.k1 * (1 - self.b + self.b * length / (self.avg_len or 1))
                score += self.idf.get(term, 0.0) * tf * (self.k1 + 1) / norm
            result.append(score)
        return result


class EmbeddingIndex:
    def __init__(self, chunks: Sequence[Chunk], embedder: Embedder):
        self.chunks = list(chunks)
        self.embedder = embedder
        self.vectors = [_normalize(v) for v in embedder.embed_documents([c.text for c in self.chunks])] if chunks else []

    def scores(self, query: str) -> list[float]:
        q = _normalize(self.embedder.embed_query(query))
        return [sum(a * b for a, b in zip(q, v)) for v in self.vectors]


def _normalize(vector: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(float(x) * float(x) for x in vector)) or 1.0
    return [float(x) / norm for x in vector]


class SentenceTransformerEmbedder:
    """Lazy wrapper around sentence-transformers (``uv sync --extra ml``)."""

    def __init__(self, model_name: str):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise DocumentError(
                "The embeddings retriever needs the ml extra: uv sync --extra ml"
            ) from exc
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:  # pragma: no cover
        return [list(map(float, v)) for v in self.model.encode(texts)]

    def embed_query(self, text: str) -> list[float]:  # pragma: no cover
        return list(map(float, self.model.encode([text])[0]))


# ---------------------------------------------------------------------- store


class DocumentStore:
    def __init__(
        self,
        folder: str | Path | None,
        chunk_size: int = 800,
        chunk_overlap: int = 120,
        retriever: str = "bm25",
        embedder: Embedder | None = None,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        self.folder = Path(folder) if folder else None
        self.chunk_size = chunk_size
        self.chunk_overlap = min(chunk_overlap, chunk_size // 2)
        self.retriever = retriever
        self.embedding_model = embedding_model
        self._embedder = embedder
        self._lock = threading.Lock()
        self._signature: tuple | None = None
        self._chunks: list[Chunk] = []
        self._infos: list[DocumentInfo] = []
        self._index: BM25Index | EmbeddingIndex | None = None
        self.retriever_note = ""

    @classmethod
    def from_settings(cls, settings, embedder: Embedder | None = None) -> "DocumentStore":
        return cls(
            settings.docs_dir,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            retriever=settings.retriever,
            embedder=embedder,
            embedding_model=settings.embedding_model,
        )

    def _files(self) -> list[Path]:
        if self.folder is None:
            raise DocumentError("No documents folder is configured. Set HELM_DOCS or [documents] path.")
        if not self.folder.is_dir():
            raise DocumentError(
                f"Documents folder not found: {self.folder}. Run `uv run helm-demo` for the demo "
                "documents, or point HELM_DOCS at your own folder."
            )
        root = self.folder.resolve()
        files = []
        for path in sorted(self.folder.rglob("*")):
            rel = path.relative_to(self.folder)
            if any(part.startswith(".") for part in rel.parts):
                continue
            if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
                continue
            if not path.resolve().is_relative_to(root):  # a symlink that points outside
                continue
            files.append(path)
        return files

    def _ensure_index(self) -> None:
        files = self._files()
        signature = tuple((str(p), p.stat().st_mtime_ns, p.stat().st_size) for p in files)
        if signature == self._signature and self._index is not None:
            return
        chunks: list[Chunk] = []
        infos: list[DocumentInfo] = []
        for path in files:
            source = path.relative_to(self.folder).as_posix()
            kind = path.suffix.lower().lstrip(".")
            try:
                pages = read_document(path)
            except Exception as exc:  # a broken file should not stop the others
                logger.warning("Could not read %s: %s", source, exc)
                infos.append(DocumentInfo(source, kind, path.stat().st_size, 0, str(exc)))
                continue
            before = len(chunks)
            index = 0
            for page, text in pages:
                for piece in chunk_text(text, self.chunk_size, self.chunk_overlap):
                    chunks.append(Chunk(source, index, piece, page))
                    index += 1
            infos.append(DocumentInfo(source, kind, path.stat().st_size, len(chunks) - before))

        self._chunks, self._infos = chunks, infos
        self._index = self._build_index(chunks)
        self._signature = signature
        logger.info("Indexed %d chunks from %d files (%s)", len(chunks), len(files), self.retriever_note)

    def _build_index(self, chunks: list[Chunk]) -> BM25Index | EmbeddingIndex:
        if self.retriever == "embeddings":
            try:
                if self._embedder is None:
                    self._embedder = SentenceTransformerEmbedder(self.embedding_model)
                self.retriever_note = "embeddings"
                return EmbeddingIndex(chunks, self._embedder)
            except Exception as exc:
                logger.warning("Embeddings unavailable, using bm25: %s", exc)
                self.retriever_note = f"bm25 (embeddings unavailable: {exc})"
                return BM25Index(chunks)
        self.retriever_note = "bm25"
        return BM25Index(chunks)

    def list_documents(self) -> list[DocumentInfo]:
        with self._lock:
            self._ensure_index()
            return list(self._infos)

    def search(self, query: str, top_k: int = 5, source_filter: str | None = None) -> list[Hit]:
        if not (query or "").strip():
            raise DocumentError("The search query is empty.")
        top_k = max(1, min(int(top_k), 20))
        with self._lock:
            self._ensure_index()
            assert self._index is not None
            scores = self._index.scores(query)
            wanted = (source_filter or "").strip().lower()
            hits = [
                Hit(chunk, score)
                for chunk, score in zip(self._chunks, scores)
                if (not wanted or wanted in chunk.source.lower())
            ]
            if isinstance(self._index, BM25Index):
                hits = [h for h in hits if h.score > 0]
                if not hits and wanted:
                    # The filter matched files but no keyword matched: return the start of those files.
                    hits = [Hit(c, 0.0) for c in self._chunks if wanted in c.source.lower() and c.index == 0]
        hits.sort(key=lambda h: (-h.score, h.chunk.source, h.chunk.index))
        return hits[:top_k]
