.PHONY: install demo ui mcp check ask eval test reset

install:
	uv sync --extra ui

demo:
	uv run helm-demo

ui:
	uv run chainlit run app.py -w

mcp:
	uv run helm-mcp

check:
	uv run helm-mcp --check

eval:
	uv run helm-eval --fresh-demo

test:
	uv run pytest -q

reset:
	rm -f data/ledger.duckdb data/vault/*_loan_app.pdf data/vault/credit_policy.md
	rm -rf .chainlit
