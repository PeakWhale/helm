.PHONY: demo reset run

reset:
	uv run python scripts/demo_reset.py

run:
	uv run chainlit run app.py -w

demo:
	uv run python scripts/demo_reset.py
	uv run chainlit run app.py -w
