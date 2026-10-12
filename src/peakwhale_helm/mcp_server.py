"""Helm MCP server.

Run it with ``helm-mcp`` (or ``python -m peakwhale_helm.mcp_server``). It speaks
MCP over stdio by default, so any MCP client (Claude Desktop, Cursor, VS Code,
the Helm chat UI) can use it.

stdio servers must never print to stdout, so all logs go to stderr.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Annotated

from pydantic import Field

from . import __version__
from .config import ConfigError, Settings, describe_settings, load_settings
from .toolbox import Toolbox, ToolError

logger = logging.getLogger("helm.mcp")

INSTRUCTIONS = """\
PeakWhale Helm gives read-only access to one database and one documents folder.

How to use it:
1. Call list_tables to see the tables, then describe_table for the columns you need.
2. Call query with ONE SELECT or WITH statement for numbers and records.
   Writes, several statements, settings and file access are blocked. Results are capped.
3. Call search_documents for facts in PDFs, Markdown or text files. Use source_filter
   to stay inside one file (for example a customer id that is part of the file name).
4. Answer only from tool results. Say which SQL query or which file and chunk each fact came from.
"""


def build_server(settings: Settings | None = None, toolbox: Toolbox | None = None):
    """Create the FastMCP server. Pass a toolbox to reuse one (tests do this)."""
    from mcp.server.fastmcp import FastMCP
    from mcp.types import ToolAnnotations

    tb = toolbox or Toolbox(settings or load_settings())
    server = FastMCP("PeakWhale Helm", instructions=INSTRUCTIONS)
    local = ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False)
    online = ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=False, openWorldHint=True)

    def list_tables() -> str:
        return tb.list_tables()

    def describe_table(
        table: Annotated[str, Field(description="Table name exactly as list_tables shows it")],
    ) -> str:
        return tb.describe_table(table)

    def query(
        sql: Annotated[str, Field(description="One read-only SQL statement that starts with SELECT or WITH")],
    ) -> str:
        return tb.query(sql)

    def list_documents() -> str:
        return tb.list_documents()

    def search_documents(
        query: Annotated[str, Field(description="What to look for, in plain words")],
        top_k: Annotated[int, Field(description="How many chunks to return (1-20)", ge=1, le=20)] = 5,
        source_filter: Annotated[
            str | None, Field(description="Only search files whose path contains this text, like a customer id")
        ] = None,
    ) -> str:
        return tb.search_documents(query, top_k=top_k, source_filter=source_filter)

    def get_stock_price(
        ticker: Annotated[str, Field(description="Stock ticker, for example AAPL")],
    ) -> str:
        return tb.get_stock_price(ticker)

    def analyze_sentiment(
        text: Annotated[str, Field(description="A short financial text, like a headline")],
    ) -> str:
        return tb.analyze_sentiment(text)

    functions = {
        "list_tables": (list_tables, local),
        "describe_table": (describe_table, local),
        "query": (query, local),
        "list_documents": (list_documents, local),
        "search_documents": (search_documents, local),
        "get_stock_price": (get_stock_price, online),
        "analyze_sentiment": (analyze_sentiment, local),
    }
    for name in tb.tool_names:
        fn, annotations = functions[name]
        server.add_tool(fn, name=name, description=tb.description(name), annotations=annotations, structured_output=False)
    server.helm_toolbox = tb  # handy for tests and embedding
    return server


def run_check(settings: Settings, out=None) -> int:
    """Print the settings and try every data source once. Returns an exit code."""
    out = out or sys.stdout
    tb = Toolbox(settings)
    print(f"PeakWhale Helm {__version__}", file=out)
    print(describe_settings(settings), file=out)
    print(f"Tools       : {', '.join(tb.tool_names)}", file=out)
    status = 0
    for label, call in (("Database", tb.list_tables), ("Documents", tb.list_documents)):
        try:
            result = call()
            print(f"\n[ok] {label}\n{result}", file=out)
        except ToolError as exc:
            status = 1
            print(f"\n[problem] {label}: {exc}", file=out)
    tb.close()
    return status


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="helm-mcp", description="PeakWhale Helm MCP server")
    parser.add_argument("--config", help="Path to a helm.toml file (default: $HELM_CONFIG or ./helm.toml)")
    parser.add_argument(
        "--transport",
        default=os.environ.get("HELM_MCP_TRANSPORT", "stdio"),
        choices=["stdio", "sse", "streamable-http"],
        help="MCP transport (default: stdio)",
    )
    parser.add_argument("--port", type=int, default=8000, help="Port for the HTTP transports")
    parser.add_argument("--check", action="store_true", help="Check the configuration and exit")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=os.environ.get("HELM_LOG_LEVEL", "INFO"),
        stream=sys.stderr,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        settings = load_settings(args.config)
    except ConfigError as exc:
        print(f"helm-mcp: configuration problem: {exc}", file=sys.stderr)
        raise SystemExit(2) from None

    if args.check:
        raise SystemExit(run_check(settings))

    server = build_server(settings)
    if args.transport != "stdio":
        server.settings.port = args.port
    logger.info("Starting Helm MCP server (%s). %s", args.transport, describe_settings(settings).replace("\n", " | "))
    server.run(transport=args.transport)


if __name__ == "__main__":
    main()
