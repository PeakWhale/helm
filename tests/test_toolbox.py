import pytest

from peakwhale_helm.market import MarketError, clean_ticker, get_stock_price
from peakwhale_helm.sentiment import SentimentAnalyzer
from peakwhale_helm.sources import Source, format_sources, unknown_citations
from peakwhale_helm.toolbox import ToolError


def test_call_checks_names_and_arguments(toolbox):
    with pytest.raises(ToolError, match="turned off"):
        toolbox.call("get_stock_price", {"ticker": "AAPL"})
    with pytest.raises(ToolError, match="Unknown tool 'rm_rf'"):
        toolbox.call("rm_rf", {})
    with pytest.raises(ToolError, match="Bad arguments for query"):
        toolbox.call("query", {"statement": "SELECT 1"})
    assert "Rows: 1" in toolbox.call("query", {"sql": "SELECT 1 AS one"})


def test_describe_and_documents_text(toolbox):
    text = toolbox.describe_table("customers")
    assert text.startswith("Table: customers (table, 100 rows)")
    assert "| risk_score | INTEGER |" in text and "Sample rows:" in text
    listing = toolbox.list_documents()
    assert "credit_policy.md" in listing and "| pdf |" in listing
    assert toolbox.search_documents("zebra").startswith('No passages matched "zebra"')


def test_clean_ticker():
    assert clean_ticker(" $aapl ") == "AAPL"
    assert clean_ticker("brk.b") == "BRK.B"
    assert clean_ticker("a;rm -rf") == "ARM-RF"


def test_stock_price_parses_stooq_and_reports_failures():
    csv_text = "Date,Open,High,Low,Close,Volume\n2026-10-08,1,1,1,99.5,1\n2026-10-09,1,1,1,101.25,1\n"
    assert get_stock_price("msft", fetch=lambda url: csv_text) == (
        "Ticker: MSFT\nPrice: $101.25\nAs of: 2026-10-09 (daily close, Stooq)"
    )

    def offline(url):
        raise OSError("network is off")

    with pytest.raises(MarketError, match="unavailable for MSFT"):
        get_stock_price("MSFT", fetch=offline)
    with pytest.raises(MarketError, match="unavailable"):
        get_stock_price("MSFT", fetch=lambda url: "<html>blocked</html>")
    with pytest.raises(MarketError, match="Missing ticker"):
        get_stock_price("  ")


def test_sentiment_with_injected_classifier():
    analyzer = SentimentAnalyzer(classifier=lambda text: {"label": "negative", "score": 0.875})
    assert analyzer.analyze("Losses widen") == "Sentiment: negative (confidence 0.88, model ProsusAI/finbert)"


def test_sources_summary_and_citations():
    query = Source("S1", "query", {"sql": "SELECT * FROM t"}, "| a |\n|---|\n| 1 |\n\nRows: 1")
    search = Source(
        "S2",
        "search_documents",
        {"query": "terms", "source_filter": "acme"},
        'Found 1 passage for "terms".\n\n[1] contracts/acme.md, chunk 2, score 1.50\nPayment terms are 45 days.',
    )
    failed = Source("S3", "query", {"sql": "DROP TABLE t"}, "Blocked: only SELECT", error=True)
    text = format_sources([query, search, failed])
    assert "- [S1] query `SELECT * FROM t` -> 1 row" in text
    assert '- [S2] search_documents "terms" in files matching "acme" -> contracts/acme.md, chunk 2' in text
    assert "- [S3] query `DROP TABLE t` -> error: Blocked: only SELECT" in text
    assert unknown_citations("A [S1], B [S7], C [S12]", [query, search]) == ["S7", "S12"]
