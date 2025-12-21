# PeakWhale™ Helm

### Helm: MCP Enabled RAG Engine for Agent to Agent Context

![Python](https://img.shields.io/badge/Python-3.11%2B-blue)
![Chainlit](https://img.shields.io/badge/Chainlit-Chat%20UI-informational)
![LangGraph](https://img.shields.io/badge/LangGraph-Orchestration-9cf)
![MCP](https://img.shields.io/badge/MCP-Tool%20Protocol-informational)
![Ollama](https://img.shields.io/badge/Ollama-Local%20LLM-lightgrey)
![DuckDB](https://img.shields.io/badge/DuckDB-Local%20Analytics%20DB-yellow)
![FAISS](https://img.shields.io/badge/FAISS-Vector%20Search-success)
![SentenceTransformers](https://img.shields.io/badge/SentenceTransformers-Embeddings-9cf)
![License: MIT](https://img.shields.io/badge/License-MIT-green)

PeakWhale™ Helm is a local first, multi agent financial intelligence system that combines structured financial data and unstructured documents to produce grounded, explainable answers in a clean, auditable workflow.

It uses MCP (Model Context Protocol) to expose tools in a consistent way, and can optionally use a local LLM (Large Language Model) via Ollama to synthesize a clear final answer from tool outputs while staying grounded.

![PeakWhale™ Helm Architecture](docs/architecture.png)

## Enterprise Business Problem

In real enterprises, critical signals are fragmented across systems.

* Transaction data lives in databases and ledgers
* Supporting evidence lives in documents like loan applications, policy forms, emails, and PDFs (Portable Document Format)
* Teams manually connect the dots, which slows decisions, increases cost, and makes outcomes harder to reproduce and audit

PeakWhale™ Helm shows how an agentic architecture can automate this end to end workflow locally, while keeping tool usage visible.

## Important Notice

All data in this repository is 100 percent synthetic.

Customer names, transaction values, merchants, and PDFs are randomly generated for demonstration purposes only.

## What PeakWhale™ Helm Does

You can ask a question like:

```text
Find the customer with the most Gambling High Risk transactions, then check their loan application for income source.
```

Behind the scenes, multiple agents collaborate to:

1. Query a financial ledger using SQL (Structured Query Language)
2. Search loan application PDFs using embeddings and vector search
3. Fetch basic market prices without API keys for demo prompts
4. Run sentiment analysis on financial text
5. Merge results into a single grounded answer, with tool calls and tool results shown in the UI (User Interface)

## System Architecture

PeakWhale™ Helm follows a supervisor style orchestration pattern using LangGraph.

High level flow:

1. User sends a message in the Chainlit chat UI (User Interface)
2. Orchestration layer (Supervisor) performs intent routing and control
3. One of four specialized agents triggers tools using MCP (Model Context Protocol)
4. Tools run against local and external demo sources and return structured results
5. Results are normalized into shared state and returned as a single response
6. Optional Ollama LLM (Large Language Model) synthesizes the final answer from tool outputs only, with guardrails to prevent hallucinated facts

Core components:

* Chainlit chat UI with tool call visibility
* Orchestration layer using LangGraph StateGraph
* MCP tool layer using FastMCP and LangChain MCP adapters
* Optional local generation via Ollama for better final response quality

Four agents:

* Quant Agent
  Queries DuckDB ledger with SQL (Structured Query Language)
  Example tool: `query_ledger`

* Researcher Agent
  Searches loan PDFs using RAG (Retrieval Augmented Generation)
  Embeddings via SentenceTransformers and a FAISS (Facebook AI Similarity Search) vector store
  Example tool: `search_loan_documents`

* Market Agent
  Fetches basic daily close prices with a free source (Stooq) and no API key
  Optional fallback to yfinance if it works on your network
  Example tool: `get_stock_price`

* Sentiment Agent
  Runs FinBERT sentiment classification for financial text
  Example tool: `analyze_sentiment`

Local data sources:

* DuckDB ledger at `data/ledger.duckdb`
* PDF vault at `data/vault/`

## Prompts To Try

Quant:

```text
Find the customer with the most Gambling High Risk transactions.
```

Researcher:

```text
Researcher: for customer 2631d00b, find the income source in the loan application.
```

Market:

```text
Get stock price for $AAPL
```

```text
Get the latest stock price for TSLA.
```

Sentiment:

```text
Sentiment: Credit risk rising as delinquencies accelerate, guidance lowered, costs up.
```

Omni-Mode (Full "Grand Slam" Analysis):

```text
Find the top high risk customer, check their loan application for income source, get the stock price for $AAPL, and analyze sentiment for 'The tech sector is showing strong resilience'.
```

This single prompt triggers all four agents in sequence:
1. **Quant**: Identifies the customer with high-risk gambling transactions.
2. **Researcher**: Looks up their specific loan PDF to find income sources.
3. **Market**: Checks live ticker prices.
4. **Sentiment**: Analyzes the provided text using FinBERT.

## Repository Structure

```text
helm/
  app.py
  src/
    graph.py
    tools.py
    sentiment_runner.py
    seed_data.py
    generate_docs.py
  scripts/
    demo_reset.py
  data/
    ledger.duckdb
    vault/
      .gitkeep
  docs/
    architecture.png
  chainlit.md
  README.md
  LICENSE
  pyproject.toml
  uv.lock
```

## Quick Start

### Prerequisites

* Python 3.11 or newer
* uv installed (Python dependency and environment manager)
* Optional: Ollama installed and running for local LLM (Large Language Model) synthesis
* Optional: DuckDB CLI if you want to browse the database manually

### Install dependencies

```bash
uv sync
```

### Reset demo data

If your Makefile includes a demo target:

```bash
make demo
```

Or run the script directly:

```bash
uv run python scripts/demo_reset.py
```

This regenerates:

* A synthetic DuckDB ledger
* Synthetic loan application PDFs

### Run the app

```bash
uv run chainlit run app.py -w
```

Then open:

```text
http://localhost:8000
```

## View Data Manually

### View the DuckDB database

```bash
duckdb data/ledger.duckdb
```

Useful queries:

```sql
SHOW TABLES;
SELECT * FROM customers LIMIT 10;
SELECT * FROM transactions LIMIT 10;
SELECT * FROM transactions WHERE category = 'Gambling/High Risk' LIMIT 25;
```

### View the PDFs

Open:

```text
data/vault/
```

Then open any generated PDF.

## Why This Project Exists

PeakWhale™ Helm demonstrates enterprise style Generative AI architecture patterns:

* Agent orchestration and routing
* Deterministic tool usage with visible traces in the UI
* Retrieval grounded answers across structured and unstructured sources
* MCP (Model Context Protocol) tool standardization for cleaner agent to agent context sharing
* Optional local LLM synthesis via Ollama, constrained to tool outputs for auditability
* Local first operation for privacy and reproducibility

## Part of the PeakWhale™ Ecosystem

* PeakWhale™ Orca, real time fraud detection
* PeakWhale™ Harbor, valuation and forecasting
* PeakWhale™ Sonar, automated threat detection
* PeakWhale™ Delta, decision support systems

## License

MIT License
© 2025 PeakWhale™

## Author

Built by Addy