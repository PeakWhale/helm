import asyncio
import sqlite3
from dataclasses import replace

from peakwhale_helm.assistant import Assistant
from peakwhale_helm.config import load_settings
from peakwhale_helm.fallback import FOOTER
from peakwhale_helm.toolbox import Toolbox
from peakwhale_helm.toolset import LocalToolSet


def ask(settings, question, **toolbox_kwargs):
    assistant = Assistant(LocalToolSet(Toolbox(settings, **toolbox_kwargs)), settings)
    return asyncio.run(assistant.answer(question))


def test_grand_slam_demo_prompt_runs_every_tool(demo_settings):
    settings = replace(demo_settings, enable_market=True, enable_sentiment=True)

    class FakeSentiment:
        def analyze(self, text):
            assert text == "The tech sector is showing strong resilience"
            return "Sentiment: positive (confidence 0.93, model fake)"

    answer, sources = ask(
        settings,
        "Find the top high risk customer, check their loan application for income source, "
        "get the stock price for $AAPL, and analyze sentiment for 'The tech sector is showing strong resilience'.",
        market_fetch=lambda url: "Date,Open,High,Low,Close,Volume\n2026-10-09,1,1,1,250.00,1\n",
        sentiment=FakeSentiment(),
    )
    assert [s.tool for s in sources] == ["query", "search_documents", "get_stock_price", "analyze_sentiment"]
    # the customer id from the SQL result is used as the document filter
    assert sources[1].args["source_filter"] == "2631d00b"
    assert "customer_id = 2631d00b, high_risk_count = 8 [S1]" in answer
    assert '"Stated Income Source: Freelance Consulting" [S2]' in answer
    assert "Price: $250.00" in answer and "[S3]" in answer
    assert "Sentiment: positive" in answer and "[S4]" in answer
    assert "**Sources**" in answer and answer.endswith(FOOTER)


def test_turned_off_tools_are_explained_not_called(demo_settings):
    answer, sources = ask(demo_settings, "Get the stock price for $TSLA and analyze sentiment of 'Sales fell'")
    assert sources == []
    assert "HELM_ENABLE_MARKET=1" in answer and "HELM_ENABLE_SENTIMENT=1" in answer


def test_saved_query_with_an_id_that_is_not_in_the_result(demo_settings):
    answer, _ = ask(demo_settings, "How many high risk transactions does customer 0000aaaa have?")
    assert "0000aaaa does not appear in the result of the saved query top_high_risk_customers" in answer


def test_unknown_question_gets_help(demo_settings):
    answer, _ = ask(demo_settings, "hello there")
    assert "could not match this question" in answer
    answer, sources = ask(demo_settings, "   ")
    assert "could not match" in answer and sources == []


def test_errors_are_shown_with_their_source(demo_settings):
    answer, sources = ask(demo_settings, "SELECT nope FROM customers")
    assert sources[0].error
    assert "Could not run the query: SQL error" in answer and "[S1]" in answer


def test_rules_work_on_any_database_not_only_the_demo(tmp_path, monkeypatch):
    db = tmp_path / "shop.sqlite"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE products (product_id TEXT, title TEXT, price REAL)")
    con.executemany(
        "INSERT INTO products VALUES (?, ?, ?)",
        [("sku100", "Blue mug", 9.5), ("sku200", "Red kettle", 39.0), ("sku300", "Tea towel", 4.25)],
    )
    con.commit()
    con.close()
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "returns_policy.md").write_text("# Returns\n\nItems can be returned within 30 days with a receipt.\n")
    monkeypatch.chdir(tmp_path)
    settings = load_settings(env={"HELM_DB": str(db), "HELM_DOCS": str(docs)})

    answer, _ = ask(settings, "How many products are there?")
    assert "The table products has 3 rows" in answer
    answer, _ = ask(settings, "What is the title of product sku200?")
    assert "title = Red kettle" in answer
    answer, _ = ask(settings, "Which columns does the products table have?")
    assert "product_id (TEXT), title (TEXT), price (REAL)" in answer
    answer, _ = ask(settings, "What does the returns policy say about receipts?")
    assert "within 30 days with a receipt" in answer and "returns_policy.md" in answer
    answer, _ = ask(settings, "SELECT title FROM products ORDER BY price DESC LIMIT 1")
    assert "Red kettle" in answer


def test_counts_with_a_filter_are_declined_not_guessed(demo_settings):
    for question in (
        "How many transactions does customer f81dbaa1 have?",
        "How many customers have a risk score above 700?",
        "How many customers joined in 2024?",
    ):
        answer, sources = ask(demo_settings, question)
        assert "can only count all rows" in answer, question
        assert " rows [S" not in answer and "query" not in [s.tool for s in sources]
    answer, _ = ask(demo_settings, "How many rows does the transactions table have?")
    assert "The table transactions has 5012 rows" in answer

