"""Small helpers to turn rows into Markdown tables and back."""

from __future__ import annotations

import datetime as _dt
import decimal
from typing import Any, Iterable, Sequence

MAX_CELL_CHARS = 200


def format_value(value: Any) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.10g}"
    if isinstance(value, decimal.Decimal):
        return format(value, "f")
    if isinstance(value, (_dt.datetime, _dt.date, _dt.time)):
        return value.isoformat(sep=" ") if isinstance(value, _dt.datetime) else value.isoformat()
    if isinstance(value, (bytes, bytearray, memoryview)):
        return f"<{len(bytes(value))} bytes>"
    text = str(value)
    text = " ".join(text.split())  # one line per cell
    if len(text) > MAX_CELL_CHARS:
        text = text[: MAX_CELL_CHARS - 3] + "..."
    return text


def _cell(value: Any) -> str:
    return format_value(value).replace("\\", "\\\\").replace("|", "\\|")


def markdown_table(columns: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    header = "| " + " | ".join(_cell(c) for c in columns) + " |"
    divider = "|" + "|".join("---" for _ in columns) + "|"
    body = ["| " + " | ".join(_cell(v) for v in row) + " |" for row in rows]
    return "\n".join([header, divider, *body])


def parse_markdown_table(text: str) -> tuple[list[str], list[list[str]]]:
    """Read the first Markdown table in ``text``. Returns (columns, rows)."""
    lines = [line.strip() for line in (text or "").splitlines()]
    start = None
    for i in range(len(lines) - 1):
        if lines[i].startswith("|") and lines[i + 1].startswith("|") and set(lines[i + 1]) <= set("|-: "):
            start = i
            break
    if start is None:
        return [], []

    def split(line: str) -> list[str]:
        inner = line.strip()
        if inner.startswith("|"):
            inner = inner[1:]
        if inner.endswith("|") and not inner.endswith("\\|"):
            inner = inner[:-1]
        cells, current, escaped = [], [], False
        for ch in inner:
            if escaped:
                current.append(ch)
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == "|":
                cells.append("".join(current).strip())
                current = []
            else:
                current.append(ch)
        cells.append("".join(current).strip())
        return cells

    columns = split(lines[start])
    rows = []
    for line in lines[start + 2 :]:
        if not line.startswith("|"):
            break
        rows.append(split(line))
    return columns, rows


def shorten(text: str, limit: int = 120) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= limit else text[: limit - 3] + "..."
