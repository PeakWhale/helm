"""The MCP server, tested the way an MCP client (Claude Desktop, Cursor, ...) sees it."""

import asyncio
import os
import sys
from dataclasses import replace

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.shared.memory import create_connected_server_and_client_session

from peakwhale_helm.mcp_server import build_server, run_check
from peakwhale_helm.toolbox import Toolbox

CORE = {"list_tables", "describe_table", "query", "list_documents", "search_documents"}


def _text(result) -> str:
    return "\n".join(block.text for block in result.content if getattr(block, "type", "") == "text")


def test_in_memory_client_sees_tools_and_gets_answers(demo_settings):
    server = build_server(toolbox=Toolbox(demo_settings))

    async def scenario():
        async with create_connected_server_and_client_session(server) as client:
            listed = await client.list_tools()
            tools = {t.name: t for t in listed.tools}
            assert set(tools) == CORE
            assert all(t.annotations.readOnlyHint for t in tools.values())
            assert "200 rows" in tools["query"].description
            assert tools["search_documents"].inputSchema["required"] == ["query"]

            result = await client.call_tool("list_tables", {})
            assert not result.isError
            assert "| transactions | table | 5012 |" in _text(result)
            assert "top_high_risk_customers" in _text(result)

            result = await client.call_tool(
                "query", {"sql": "SELECT name FROM customers WHERE customer_id = '2631d00b'"}
            )
            assert "Demo Tiffany Miller" in _text(result)

            result = await client.call_tool("query", {"sql": "DROP TABLE customers"})
            assert result.isError and "Blocked" in _text(result)

            result = await client.call_tool("search_documents", {"query": "income source", "source_filter": "2631d00b"})
            text = _text(result)
            assert "[1] 2631d00b_loan_app.pdf, chunk 0 (page 1)" in text
            assert "Freelance Consulting" in text

            result = await client.call_tool("describe_table", {"table": "nope"})
            assert result.isError and "Known tables" in _text(result)

    asyncio.run(scenario())


def test_optional_tools_are_only_listed_when_turned_on(demo_settings):
    settings = replace(demo_settings, enable_market=True, enable_sentiment=True)

    class FakeSentiment:
        def analyze(self, text):
            return f"Sentiment: positive (confidence 0.90) for {len(text)} chars"

    toolbox = Toolbox(
        settings,
        market_fetch=lambda url: "Date,Open,High,Low,Close,Volume\n2026-10-09,1,2,0.5,123.45,100\n",
        sentiment=FakeSentiment(),
    )
    server = build_server(toolbox=toolbox)

    async def scenario():
        async with create_connected_server_and_client_session(server) as client:
            names = {t.name for t in (await client.list_tools()).tools}
            assert names == CORE | {"get_stock_price", "analyze_sentiment"}
            price = _text(await client.call_tool("get_stock_price", {"ticker": "$aapl"}))
            assert "Ticker: AAPL" in price and "$123.45" in price and "2026-10-09" in price
            mood = _text(await client.call_tool("analyze_sentiment", {"text": "Profits beat forecasts"}))
            assert mood.startswith("Sentiment: positive")

    asyncio.run(scenario())


def test_stdio_handshake_with_the_real_entry_point(demo_dir, tmp_path):
    """Start `python -m peakwhale_helm.mcp_server` like an MCP client does, over stdio."""
    env = {
        **os.environ,
        "HELM_DB": str(demo_dir / "ledger.duckdb"),
        "HELM_DOCS": str(demo_dir / "vault"),
        "HELM_LOG_LEVEL": "WARNING",
    }
    params = StdioServerParameters(
        command=sys.executable, args=["-m", "peakwhale_helm.mcp_server"], env=env, cwd=str(tmp_path)
    )

    async def scenario():
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                init = await session.initialize()
                assert init.serverInfo.name == "PeakWhale Helm"
                assert "read-only" in (init.instructions or "")
                names = {t.name for t in (await session.list_tools()).tools}
                assert names == CORE
                result = await session.call_tool("query", {"sql": "SELECT COUNT(*) AS n FROM customers"})
                assert "| 100 |" in _text(result)

    asyncio.run(asyncio.wait_for(scenario(), timeout=60))


def test_check_command_reports_problems(tmp_path, monkeypatch, demo_settings, capsys):
    assert run_check(demo_settings) == 0
    out = capsys.readouterr().out
    assert "[ok] Database" in out and "[ok] Documents" in out
    broken = demo_settings.with_data(tmp_path / "missing.duckdb", tmp_path / "missing-docs")
    assert run_check(broken) == 1
    out = capsys.readouterr().out
    assert "[problem] Database" in out and "helm-demo" in out
