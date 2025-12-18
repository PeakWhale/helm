# PeakWhale™ Helm

### Multi Agent Financial Intelligence System (Local First, Open Source, Demo First)

![Python](https://img.shields.io/badge/Python-3.11%2B-blue)
![Chainlit](https://img.shields.io/badge/Chainlit-UI%20%26%20Observability-informational)
![DuckDB](https://img.shields.io/badge/DuckDB-Local%20Analytics%20DB-yellow)
![FAISS](https://img.shields.io/badge/FAISS-Vector%20Search-success)
![SentenceTransformers](https://img.shields.io/badge/SentenceTransformers-Embeddings-9cf)
![Ollama](https://img.shields.io/badge/Ollama-Local%20LLM%20Runtime-lightgrey)
![License: MIT](https://img.shields.io/badge/License-MIT-green)

PeakWhale™ Helm is a local first, multi agent AI system that demonstrates how enterprises can combine structured financial data and unstructured documents to produce grounded, explainable answers in a clean, auditable workflow.

![PeakWhale™ Helm Architecture](docs/architecture.png)

If the architecture image does not render on GitHub, confirm the file exists at `docs/architecture.png` and is committed to the repository.

## Enterprise Business Problem

In real enterprises, critical signals are fragmented across systems.

* Transaction data lives in databases and ledgers
* Supporting evidence lives in documents like loan applications, policy forms, emails, and PDFs
* Teams manually connect the dots, which slows decisions, increases cost, and makes outcomes harder to reproduce and audit

PeakWhale™ Helm shows how an agentic architecture can automate this end to end workflow locally, while keeping reasoning explainable and tool usage visible.

## Important Notice

All data in this repository is 100% synthetic.
Customer names, transaction values, merchants, and PDFs are randomly generated for demonstration purposes only.

## What PeakWhale™ Helm Does

You can ask a question like:

```text
Find the customer with the most Gambling High Risk transactions, then check their loan application for income source.
```

Behind the scenes, multiple agents collaborate to:

1. Query a financial ledger using SQL
2. Search loan application PDFs using embeddings and vector search
3. Merge results into a single grounded answer
4. Explain how the answer was derived

All locally, without cloud APIs.

## System Architecture

PeakWhale™ Helm is built around a simple orchestration pattern. A Supervisor agent routes the request to specialized agents and tools, then combines the results into a final answer.

Key components:

* Chainlit UI provides the chat experience and tool call visibility
* Supervisor Agent routes intent and orchestrates the flow
* Quant Agent performs SQL analysis over DuckDB
* Researcher Agent performs semantic search over loan PDFs
* Extractor normalizes facts into shared state
* DuckDB Ledger is the local transaction database
* PDF Vault stores synthetic loan documents
* FAISS provides the vector index for retrieval
* Local LLM via Ollama provides reasoning and response generation

## Example End to End Flow

### User prompt

```text
Find the customer with the most Gambling High Risk transactions, then check their loan application for income source.
```

### Execution

1. Supervisor detects a multi step request
2. Quant Agent queries DuckDB to find the top customer
3. Researcher Agent searches the PDF vault for that customer’s loan application details
4. Extractor merges structured and unstructured facts into shared state
5. The LLM produces a grounded answer, and Chainlit shows tool calls and outputs

### Example answer

```text
Customer ID: 2631d00b
High Risk transactions: 8
Income source: Freelance Consulting
```

## Tech Stack

* Python 3.11+
* DuckDB, local analytical database
* Chainlit, UI and logging
* FAISS, vector search
* SentenceTransformers all MiniLM L6 v2 embeddings
* Ollama, local LLM runtime
* UV, Python environment and dependency management

## Repository Structure

```text
peakwhale-helm/
  app.py
  main.py
  src/
    graph.py
    tools.py
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
  README.md
  LICENSE
  pyproject.toml
  uv.lock
```

## Quick Start

### Prerequisites

* Python 3.11 or newer
* UV installed
* Ollama installed and running
* DuckDB installed if you want the CLI viewer

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

Open the DuckDB shell against the local file:

```bash
duckdb data/ledger.duckdb
```

Useful commands:

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

Then double click any PDF to view it.

## Why This Project Exists

PeakWhale™ Helm demonstrates enterprise style AI architecture patterns:

* Agent orchestration and routing
* Deterministic tool usage
* Retrieval grounded answers
* Clear separation of concerns
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
AI Systems, Agentic Architectures, Enterprise ML
