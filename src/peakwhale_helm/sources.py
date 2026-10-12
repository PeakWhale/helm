"""Sources: a record of every tool call behind an answer.

Each tool result gets a tag such as ``[S1]``. Answers cite these tags, and the
"Sources" list at the end of an answer says which tool, which SQL and how many
rows, or which file and chunk, each tag points to.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .formatting import shorten

_ROWS = re.compile(r"^Rows: (\d+)(.*)$", re.MULTILINE)
_HIT = re.compile(r"^\[(\d+)\] (.+?), score -?[\d.]+$", re.MULTILINE)
_CITE = re.compile(r"\[S(\d+)\]")


@dataclass
class Source:
    id: str
    tool: str
    args: dict[str, Any] = field(default_factory=dict)
    output: str = ""
    error: bool = False

    def hits(self) -> list[str]:
        """Labels like 'file.pdf, chunk 0 (page 1)' for a search_documents result."""
        return [m.group(2) for m in _HIT.finditer(self.output)]

    def row_count(self) -> int | None:
        match = _ROWS.search(self.output)
        return int(match.group(1)) if match else None

    def summary(self) -> str:
        args = self.args or {}
        if self.tool == "query":
            text = f"query `{shorten(str(args.get('sql', '')), 110)}`"
            rows = self.row_count()
            if rows is not None and not self.error:
                text += f" -> {rows} row{'s' if rows != 1 else ''}"
                if "row limit" in self.output:
                    text += " (cut at the row limit)"
        elif self.tool == "search_documents":
            text = f'search_documents "{shorten(str(args.get("query", "")), 60)}"'
            if args.get("source_filter"):
                text += f' in files matching "{args["source_filter"]}"'
            hits = self.hits()
            if hits and not self.error:
                more = f" and {len(hits) - 3} more" if len(hits) > 3 else ""
                text += " -> " + "; ".join(hits[:3]) + more
            elif not self.error:
                text += " -> no match"
        elif self.tool == "describe_table":
            text = f"describe_table {args.get('table', '')}"
        elif self.tool == "get_stock_price":
            text = f"get_stock_price {args.get('ticker', '')}"
        elif self.tool == "analyze_sentiment":
            text = f'analyze_sentiment "{shorten(str(args.get("text", "")), 60)}"'
        else:
            text = self.tool
        if self.error:
            text += f" -> error: {shorten(self.output, 140)}"
        return text


def error_text(exc: BaseException) -> str:
    """A clean one-line error message from a tool failure (local or over MCP)."""
    text = str(exc).strip() or type(exc).__name__
    text = re.sub(r"^(?:Error executing tool \w+: |Error: )", "", text)
    return text


def format_sources(sources: list[Source]) -> str:
    if not sources:
        return ""
    lines = ["**Sources**"]
    lines += [f"- [{s.id}] {s.summary()}" for s in sources]
    return "\n".join(lines)


def cited_ids(text: str) -> set[str]:
    return {f"S{n}" for n in _CITE.findall(text or "")}


def unknown_citations(text: str, sources: list[Source]) -> list[str]:
    known = {s.id for s in sources}
    return sorted(cited_ids(text) - known, key=lambda x: int(x[1:]))


@dataclass
class Event:
    """One thing that happened while answering, for UIs and logs.

    kind is one of: "tool_call", "tool_result", "note", "answer".
    """

    kind: str
    text: str = ""
    tool: str | None = None
    args: dict[str, Any] | None = None
    source: Source | None = None
