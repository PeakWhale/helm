# PeakWhale™ Helm

PeakWhale™ Helm is a local first multi agent demo that answers questions by routing work to the right specialist.

It combines
1. A Quant Agent for structured data via SQL (Structured Query Language) on DuckDB
2. A Researcher Agent for unstructured data via RAG (Retrieval Augmented Generation) over loan application PDFs
3. A Supervisor Agent that routes the user request to the right agent and merges results into a final answer

This project is built to be easy to run on a laptop and easy to explain in interviews.

## Architecture

![PeakWhale™ Helm Architecture](docs/architecture.png)

## What this demo does

Example question
Find the customer with the most Gambling High Risk transactions. Then check their loan application for income source.

What happens
1. Supervisor routes to Quant Agent for the database question
2. Quant Agent runs a SQL query on DuckDB and returns the top customer id
3. Supervisor routes to Researcher Agent for the PDF question
4. Researcher Agent searches the PDF vault using embeddings and returns the relevant excerpt
5. Supervisor merges both results into a clean final response

## Prerequisites

1. macOS
2. uv installed
3. Ollama installed and running
4. A model pulled in Ollama, for example llama3.2

Confirm Ollama is running
```bash
ollama list
