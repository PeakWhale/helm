````md
# ⚓ PeakWhale™ Helm
### Multi-Agent Financial Intelligence System (Local • Open Source • Demo First)

PeakWhale™ Helm is a local first multi agent AI system that demonstrates how agentic architectures can analyze structured financial data and unstructured documents together in a clean explainable enterprise style workflow.

This project is part of the PeakWhale™ open source ecosystem built to showcase real world AI architecture patterns for recruiters engineers and practitioners.

## Important Notice
All data in this repository is 100 percent synthetic.  
Customer names transaction values merchants documents and PDFs are randomly generated and used for demonstration purposes only.

## What PeakWhale™ Helm Does

In real enterprises insights are fragmented.

- Transactions live in databases  
- Supporting evidence lives in documents  
- Analysts manually connect the dots  

PeakWhale™ Helm shows how AI agents can do this automatically.

Example prompt you can ask:

```text
Find the customer with the most Gambling High Risk transactions and check their loan application for income source.
````

Behind the scenes multiple agents collaborate to:

1. Query a financial ledger using SQL which stands for Structured Query Language
2. Search loan application PDFs using embeddings and vector search
3. Merge results into a single grounded answer
4. Explain how the answer was derived

All locally without cloud APIs.

## System Architecture

![PeakWhale™ Helm Architecture](docs/peakwhale-helm-architecture.png)

### Architecture Overview

| Component                                                  | Responsibility                                           |
| ---------------------------------------------------------- | -------------------------------------------------------- |
| Chainlit UI                                                | Interactive chat interface and step logging              |
| Supervisor Agent                                           | Routes intent and orchestrates agents                    |
| Quant Agent                                                | Executes SQL queries on DuckDB                           |
| Researcher Agent                                           | Searches loan PDFs via semantic search                   |
| Extractor                                                  | Normalizes facts into shared state                       |
| DuckDB                                                     | Local analytical database for transactions and customers |
| PDF Vault                                                  | Synthetic loan applications stored as PDFs               |
| FAISS which stands for Facebook AI Similarity Search       | Vector index for document retrieval                      |
| Local LLM which stands for Large Language Model via Ollama | Reasoning and response generation                        |

## Example End to End Flow

### User prompt

```text
Find the customer with the most Gambling High Risk transactions and check their loan application for income source.
```

### Agent execution flow

1. Supervisor detects a multi step request
2. Quant Agent queries DuckDB to identify the top risk customer
3. Researcher Agent searches PDFs for that customer and extracts the requested line
4. Extractor merges structured and unstructured results into shared state
5. The local LLM produces a clear grounded response

### Example final answer

```text
Customer ID: 2631d00b
High Risk transactions: 8
Income source: Freelance Consulting
```

## Technology Stack

* Python 3.11
* DuckDB local analytical database
* Chainlit agent user interface and logging
* LangGraph style orchestration
* FAISS which stands for Facebook AI Similarity Search
* SentenceTransformers MiniLM for embeddings
* Ollama for local LLM runtime
* uv for Python environment management

## Repository Structure

```text
peakwhale-helm/
├── app.py
├── main.py
├── src/
│   ├── graph.py
│   ├── tools.py
│   ├── seed_data.py
│   └── generate_docs.py
├── scripts/
│   └── demo_reset.py
├── data/
│   ├── ledger.duckdb
│   └── vault/
├── docs/
│   └── peakwhale-helm-architecture.png
├── README.md
├── LICENSE
└── pyproject.toml
```

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

If you have a Makefile target:

```bash
make demo
```

Or run the script directly:

```bash
uv run python scripts/demo_reset.py
```

This deletes old demo data and regenerates:

* Synthetic ledger in DuckDB
* Synthetic loan application PDFs

### Run the application

```bash
uv run chainlit run app.py -w
```

Open:

```text
http://localhost:8000
```

## Viewing Data Manually

### View DuckDB in the official CLI

```bash
duckdb data/ledger.duckdb
```

Then run:

```sql
SHOW TABLES;
SELECT * FROM customers LIMIT 10;
SELECT * FROM transactions LIMIT 10;
SELECT * FROM transactions WHERE category = 'Gambling/High Risk' LIMIT 25;
```

### View loan application PDFs

Open this folder in Finder:

```text
data/vault/
```

Then double click any PDF.

## Why This Project Exists

PeakWhale™ Helm demonstrates:

* Agent to agent coordination
* Deterministic tool usage
* Explainable reasoning paths with tool traces
* Local first AI architecture
* Enterprise style separation of concerns

This project is intentionally interview ready not a toy chatbot.

## Part of the PeakWhale™ Ecosystem

* PeakWhale™ Orca real time fraud detection
* PeakWhale™ Harbor real estate valuation
* PeakWhale™ Sonar automated threat detection
* PeakWhale™ Delta decision support systems

## License

MIT License
© 2025 PeakWhale™

## Author

Built by Addy
AI systems agentic architectures enterprise ML

If you are reviewing this repository start with the architecture diagram and run the demo.

```
::contentReference[oaicite:0]{index=0}
```
