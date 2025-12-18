.PHONY: demo ui seed docs reset

demo:
	uv run python scripts/demo_reset.py

ui:
	uv run chainlit run app.py -w

seed:
	uv run python src/seed_data.py

docs:
	uv run python src/generate_docs.py

reset:
	rm -f data/ledger.duckdb
	rm -f data/vault/*_loan_app.pdf
	rm -rf .chainlit __pycache__ src/__pycache__
	mkdir -p data/vault
