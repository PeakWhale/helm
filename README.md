````markdown
# ⚓ PeakWhale™ Helm  
### Multi Agent Financial Intelligence System (Local · Open Source · Demo First)

PeakWhale™ Helm is a local first, multi agent AI system that demonstrates how agentic architectures can analyze structured financial data and unstructured documents together in a clean, explainable, enterprise style workflow.

This project is part of the PeakWhale™ open source ecosystem, built to showcase real world AI architecture patterns for recruiters, engineers, and practitioners.

## Important Notice
All data in this repository is 100 percent synthetic.  
Customer names, transaction values, merchants, documents, and Portable Document Format (PDF) files are randomly generated and used for demonstration purposes only.

## What PeakWhale™ Helm Does

In real enterprises, insights are fragmented:

* Transactions live in databases  
* Supporting evidence lives in documents  
* Analysts manually connect the dots  

PeakWhale™ Helm shows how AI agents can automate that workflow.

Example user question:

```text
Find the customer with the most Gambling High Risk transactions and check their loan application for income source.
````

Behind the scenes, multiple agents collaborate to:

1. Query a financial ledger using Structured Query Language (SQL)
2. Search loan application PDFs using embeddings and vector search
3. Merge results into a single, grounded answer
4. Explain how the answer was derived

All locally, without cloud APIs.

## System Architecture

![PeakWhale™ Helm Architecture](docs/peakwhale-helm-architecture.png)

| Component                                   | Responsibility                                      |
| ------------------------------------------- | --------------------------------------------------- |
| Chainlit UI                                 | Interactive chat interface and clean step logging   |
| Supervisor Agent                            | Routes intent and orchestrates agent execution      |
| Quant Agent                                 | Executes SQL queries against DuckDB                 |
| Researcher Agent                            | Searches loan PDFs via semantic retrieval           |
| Shared State Extractor                      | Normalizes key facts into a shared state object     |
| DuckDB                                      | Local analytical database for the ledger            |
| PDF Vault                                   | Local folder containing synthetic loan applications |
| Facebook AI Similarity Search (FAISS)       | Vector index for document retrieval                 |
| Local Large Language Model (LLM) via Ollama | Reasoning, planning, and final response writing     |

## Example End to End Flow

### User Prompt

```text
Find the customer with the most Gambling High Risk transactions and check their loan application for income source.
```

### Agent Execution Flow

1. Supervisor detects a multi step request
2. Quant queries DuckDB to identify the top customer by high risk transaction count
3. Researcher searches the matching customer’s loan PDF for the income source field
4. Extractor merges structured and unstructured results into shared state
5. The LLM generates a grounded answer

### Example Final Answer

Customer ID: 2631d00b
High Risk transactions: 8
Income source: Freelance Consulting

## Technology Stack

* Python 3.11
* DuckDB (local analytical database)
* Chainlit (agent UI and step logging)
* Graph based orchestration (LangGraph style)
* Facebook AI Similarity Search (FAISS) for vector search
* SentenceTransformers MiniLM for embeddings
* Ollama for local LLM runtime
* uv for Python environment management

## Repository Structure

* app.py
* main.py
* src/

  * graph.py
  * tools.py
  * seed_data.py
  * generate_docs.py
* scripts/

  * demo_reset.py
* data/

  * ledger.duckdb
  * vault/
* docs/

  * peakwhale-helm-architecture.png
* README.md
* LICENSE
* pyproject.toml
* uv.lock

## Quick Start

### Prerequisites

* Python 3.11 or newer
* Ollama installed and running
* DuckDB installed
* uv installed

### Install dependencies

```bash
uv sync
```

### Reset demo data

```bash
make demo
```

This clears and regenerates:

* Synthetic ledger data in DuckDB
* Synthetic loan application PDFs in the vault

### Run the application

```bash
uv run chainlit run app.py
```

Then open:

```text
http://localhost:8000
```

## Viewing Data Manually

### View DuckDB in the official CLI (Command Line Interface)

```bash
duckdb data/ledger.duckdb
```

Inside DuckDB, run:

```sql
SHOW TABLES;
SELECT * FROM customers LIMIT 10;
SELECT * FROM transactions WHERE category = 'Gambling/High Risk';
```

### View loan PDFs

Open the folder:

```text
data/vault/
```

Then double click any PDF.

## Why This Project Exists

PeakWhale™ Helm demonstrates:

* Agent to agent coordination
* Deterministic tool usage
* Explainable reasoning paths
* Local first AI architecture
* Enterprise style separation of concerns

This project is intentionally interview ready, not a toy chatbot.

## Part of the PeakWhale™ Ecosystem

* PeakWhale™ Orca: Real time fraud detection
* PeakWhale™ Harbor: Real estate valuation
* PeakWhale™ Sonar: Automated threat detection
* PeakWhale™ Delta: Decision support systems

## License

MIT License
© 2025 PeakWhale™

## Author

Built by Addy (Aditya Rao Kaveti)
AI Systems · Agentic Architectures · Enterprise ML

If you are reviewing this repository, start with the architecture diagram and run the demo.

```
::contentReference[oaicite:0]{index=0}
```
