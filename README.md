# PeakWhale™ Helm
### Enterprise Financial Intelligence & Multi-Agent Orchestration

![Python](https://img.shields.io/badge/Python-3.11%2B-blue)
![LangGraph](https://img.shields.io/badge/LangGraph-Orchestration-9cf)
![MCP](https://img.shields.io/badge/MCP-Protocol-informational)
![Ollama](https://img.shields.io/badge/Local%20LLM-Ollama-lightgrey)
![License: MIT](https://img.shields.io/badge/License-MIT-green)

**PeakWhale™ Helm** is a local-first, agentic financial intelligence system designed to demonstrate the future of enterprise automation. It combines structured financial data (SQL ledgers) with unstructured evidence (loan documents) to produce grounded, explainable insights.

By leveraging the **Model Context Protocol (MCP)** and **LangGraph** orchestration, Helm enables four specialized agents to collaborate seamlessly, executing complex research tasks that previously required human analysis.

---

> [!NOTE]
> **Synthetic Data Only**: All customer names, transaction values, and documents in this repository are randomly generated for demonstration purposes.

## The Enterprise Challenge

In modern banking, critical risk signals are fragmented:
* **Transaction Red Flags** live in core banking databases (SQL).
* **Income Evidence** lives in unstructured documents (PDFs).
* **Market Context** lives in external feeds.

Human analysts must manually connect these dots, leading to slow decisions and audit gaps. **Helm** automates this end-to-end workflow locally, ensuring privacy and reproducibility.

![PeakWhale™ Helm Architecture](docs/architecture.png)

## Core Capabilities

### 1. Multi-Agent Collaboration ("Omni-Mode")
Helm's primary innovation is its ability to decompose a complex natural language request into a deterministic execution plan. 

**Try the "Grand Slam" Prompt:**
```text
Find the top high risk customer, check their loan application for income source, get the stock price for $AAPL, and analyze sentiment for 'The tech sector is showing strong resilience'.
```

**What happens behind the scenes:**
1.  **Quant Agent**: Queries the DuckDB ledger to identify the highest-risk customer (`2631d00b`) with gambling activity.
2.  **Research Agent**: Retrieves the specific loan application PDF for that customer and extracts income details using RAG.
3.  **Market Agent**: Fetches real-time equity pricing for `$AAPL`.
4.  **Sentiment Agent**: Runs a local FinBERT model to score the provided market commentary.

The system then synthesizes these distinct data points into a single, cohesive research report.

### 2. Specialized Autonomous Agents

| Agent | Function | Technology |
| :--- | :--- | :--- |
| **Quant** | Forensic accounting & ledger analysis | **DuckDB** + SQL Generation |
| **Researcher** | Document intelligence & fact-checking | **SentenceTransformers** + **FAISS** (RAG) |
| **Market** | Real-time equity & asset pricing | **Stooq** / **yfinance** |
| **Sentiment** | Financial news tone analysis | **FinBERT** (Transformer) |

## System Architecture

Helm follows a **Supervisor-Worker** orchestration pattern:
1.  **Orchestration**: The Supervisor analyzes user intent and routes tasks to the appropriate worker agents.
2.  **Execution**: Agents execute tools via MCP, ensuring strict separation of concerns.
3.  **Synthesis**: Results are aggregated into a shared state object.
4.  **Generation**: An optional local LLM (Ollama) generates the final natural language response based *only* on the verified tool outputs.

## Quick Start

### Prerequisites
*   **Python 3.11+**
*   **uv** (recommended for ultra-fast dependency management)
*   *(Optional)* **Ollama** running `llama3.2` for full synthesized responses.

### Installation

```bash
# 1. Install dependencies
uv sync

# 2. Generate synthetic enterprise data
uv run python scripts/demo_reset.py

# 3. Launch the Agent UI
uv run chainlit run app.py -w
```
Access the dashboard at `http://localhost:8000`.

## Repository Structure

```text
helm/
├── app.py                  # Chainlit UI entry point
├── src/
│   ├── graph.py            # LangGraph orchestration logic
│   ├── tools.py            # MCP Tool definitions
│   ├── sentiment_runner.py # Persistent inference server (FinBERT)
│   └── seed_data.py        # Synthetic data generator
├── data/
│   ├── ledger.duckdb       # Local SQL database
│   └── vault/              # Encrypted document store (PDFs)
└── README.md
```

## Why This Project Exists

PeakWhale™ Helm demonstrates production-grade Generative AI patterns for regulated industries:
*   **Deterministic Tool Usage**: Every data point is traceable to a specific tool call.
*   **Privacy-First Architecture**: Runs 100% locally with no data egress.
*   **Standardized Context**: Uses MCP to decouple agents from the orchestration layer.

---

## Part of the PeakWhale™ Ecosystem
*   **PeakWhale™ Orca**: Real-time fraud detection.
*   **PeakWhale™ Harbor**: Valuation and forecasting.
*   **PeakWhale™ Sonar**: Automated threat detection.

## License
MIT License © 2025 PeakWhale™

## Author
Built by Addy