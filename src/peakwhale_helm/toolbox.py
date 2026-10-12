"""The Helm tools, as plain Python functions that return text.

The same functions back the MCP server, the LLM agent and the no-LLM fallback,
so every path sees the same data and the same safety rules.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

from .config import Settings, redact_url
from .database import Database, DatabaseError
from .documents import DocumentError, DocumentStore
from .formatting import markdown_table
from .market import Fetcher, MarketError, get_stock_price
from .sentiment import SentimentAnalyzer, SentimentError

logger = logging.getLogger("helm.tools")

CORE_TOOLS = ("list_tables", "describe_table", "query", "list_documents", "search_documents")
OPTIONAL_TOOLS = ("get_stock_price", "analyze_sentiment")

TOOL_DESCRIPTIONS = {
    "list_tables": (
        "List the tables in the connected database with their row counts and the SQL dialect. "
        "Start here to learn what data exists. Also shows saved, reviewed SQL queries."
    ),
    "describe_table": (
        "Show the columns, column types and a few sample rows of one table. "
        "Use it before writing SQL so you use the right column names."
    ),
    "query": (
        "Run ONE read-only SQL query (SELECT or WITH) on the connected database and get the "
        "result as a Markdown table. The result is capped at {max_rows} rows, so prefer "
        "COUNT, SUM and GROUP BY over pulling many rows. Writes, several statements, settings "
        "and file access are blocked."
    ),
    "list_documents": "List the files in the documents folder (PDF, Markdown, TXT) and how many chunks each has.",
    "search_documents": (
        "Search the documents folder and return the best matching text chunks. Each chunk shows its "
        "source file and chunk number, so you can cite it. Use source_filter to search only files whose "
        "path contains a word or an id (for example a customer id)."
    ),
    "get_stock_price": (
        "Get the latest daily closing price for a stock ticker such as AAPL. This calls the internet (Stooq)."
    ),
    "analyze_sentiment": (
        "Score the financial sentiment (positive, negative or neutral) of a short text with the FinBERT model."
    ),
}


class ToolError(Exception):
    """A tool could not answer. The message tells the caller what to fix."""


class Toolbox:
    def __init__(
        self,
        settings: Settings,
        *,
        database: Database | None = None,
        documents: DocumentStore | None = None,
        market_fetch: Fetcher | None = None,
        sentiment: SentimentAnalyzer | None = None,
    ):
        self.settings = settings
        self.database = database or Database.from_settings(settings)
        self.documents = documents or DocumentStore.from_settings(settings)
        self._market_fetch = market_fetch
        self._sentiment = sentiment

    # ------------------------------------------------------------------ catalogue

    @property
    def tool_names(self) -> list[str]:
        names = list(CORE_TOOLS)
        if self.settings.enable_market:
            names.append("get_stock_price")
        if self.settings.enable_sentiment:
            names.append("analyze_sentiment")
        return names

    def description(self, name: str) -> str:
        return TOOL_DESCRIPTIONS[name].format(max_rows=self.settings.max_rows)

    def function(self, name: str) -> Callable[..., str]:
        if name not in self.tool_names:
            available = ", ".join(self.tool_names)
            if name in OPTIONAL_TOOLS:
                raise ToolError(f"The tool {name} is turned off in this setup. Available tools: {available}")
            raise ToolError(f"Unknown tool {name!r}. Available tools: {available}")
        return getattr(self, name)

    def call(self, name: str, args: dict[str, Any] | None = None) -> str:
        func = self.function(name)
        try:
            return func(**(args or {}))
        except TypeError as exc:
            raise ToolError(f"Bad arguments for {name}: {exc}") from None

    # ------------------------------------------------------------------ database tools

    def list_tables(self) -> str:
        try:
            tables = self.database.list_tables()
        except DatabaseError as exc:
            raise ToolError(str(exc)) from None
        where = self.database.target if self.database.kind == "postgres" else Path(self.database.target).name
        lines = [f"Database: {self.database.kind} ({redact_url(where)})", f"SQL dialect: {self.database.dialect}", ""]
        if tables:
            rows = [(t.name, t.kind, "?" if t.rows is None else t.rows) for t in tables]
            lines.append(markdown_table(["table", "type", "rows"], rows))
        else:
            lines.append("(no tables found)")
        if self.settings.saved_queries:
            lines += ["", "Saved queries (reviewed SQL you can pass to `query`):"]
            for saved in self.settings.saved_queries:
                about = f": {saved.description}" if saved.description else ""
                lines.append(f"- {saved.name}{about}\n  SQL: {' '.join(saved.sql.split())}")
        return "\n".join(lines)

    def describe_table(self, table: str) -> str:
        try:
            info, columns, sample = self.database.describe_table(table)
        except DatabaseError as exc:
            raise ToolError(str(exc)) from None
        rows = "" if info.rows is None else f", {info.rows} rows"
        parts = [
            f"Table: {info.name} ({info.kind}{rows})",
            "",
            markdown_table(["column", "type"], columns),
            "",
            "Sample rows:",
            sample.to_markdown().rsplit("\n\nRows:", 1)[0],
        ]
        return "\n".join(parts)

    def query(self, sql: str) -> str:
        try:
            return self.database.query(sql).to_markdown()
        except DatabaseError as exc:
            raise ToolError(str(exc)) from None

    # ------------------------------------------------------------------ document tools

    def list_documents(self) -> str:
        try:
            docs = self.documents.list_documents()
        except DocumentError as exc:
            raise ToolError(str(exc)) from None
        if not docs:
            return f"No PDF, Markdown or TXT files in the documents folder ({self.documents.folder.name})."
        rows = [(d.source, d.kind, d.chunks, d.error or "") for d in docs]
        return f"Documents folder: {self.documents.folder.name}\n\n" + markdown_table(
            ["file", "type", "chunks", "problem"], rows
        )

    def search_documents(self, query: str, top_k: int = 5, source_filter: str | None = None) -> str:
        try:
            hits = self.documents.search(query, top_k=top_k, source_filter=source_filter)
        except DocumentError as exc:
            raise ToolError(str(exc)) from None
        where = f', filter: "{source_filter}"' if source_filter else ""
        note = self.documents.retriever_note or self.documents.retriever
        if not hits:
            hint = " Try other words, or remove the filter." if source_filter else " Try other words."
            return f'No passages matched "{query}" (retriever: {note}{where}).{hint}'
        noun = "passage" if len(hits) == 1 else "passages"
        parts = [f'Found {len(hits)} {noun} for "{query}" (retriever: {note}{where}).']
        for n, hit in enumerate(hits, start=1):
            parts.append(f"\n[{n}] {hit.chunk.label}, score {hit.score:.2f}\n{hit.chunk.text}")
        return "\n".join(parts)

    # ------------------------------------------------------------------ optional tools

    def get_stock_price(self, ticker: str) -> str:
        try:
            return get_stock_price(ticker, fetch=self._market_fetch)
        except MarketError as exc:
            raise ToolError(str(exc)) from None

    def analyze_sentiment(self, text: str) -> str:
        if self._sentiment is None:
            self._sentiment = SentimentAnalyzer()
        try:
            return self._sentiment.analyze(text)
        except SentimentError as exc:
            raise ToolError(str(exc)) from None

    def close(self) -> None:
        self.database.close()
