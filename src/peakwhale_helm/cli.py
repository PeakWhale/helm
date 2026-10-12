"""``helm-ask``: ask one question from the terminal.

    uv run helm-ask "Which customer has the most high risk transactions?"
    uv run helm-ask --no-llm "What tables are in the database?"
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys

from .assistant import Assistant
from .config import ConfigError, Settings, load_settings
from .formatting import shorten
from .toolbox import Toolbox
from .toolset import LocalToolSet


def make_assistant(settings: Settings, use_llm: bool = True) -> Assistant:
    """Assistant with in-process tools, and the configured LLM when it is available."""
    toolset = LocalToolSet(Toolbox(settings))
    if not use_llm:
        return Assistant(toolset, settings, llm_problem="LLM not used (--no-llm).")
    from .llm import build_chat_model

    chat = build_chat_model(settings.llm)
    return Assistant(toolset, settings, llm=chat.model, llm_label=chat.label, llm_problem=chat.problem)


async def _run(assistant: Assistant, question: str, verbose: bool) -> str:
    answer = ""
    async for event in assistant.ask(question):
        if event.kind == "tool_call" and verbose:
            print(f"-> {event.tool} {event.args}", file=sys.stderr)
        elif event.kind == "tool_result" and verbose:
            print(f"<- {event.tool}: {shorten(event.text, 160)}", file=sys.stderr)
        elif event.kind == "note":
            print(f"note: {event.text}", file=sys.stderr)
        elif event.kind == "answer":
            answer = event.text
    return answer


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="helm-ask", description="Ask PeakWhale Helm one question")
    parser.add_argument("question", nargs="+", help="Your question, or one SELECT query")
    parser.add_argument("--config", help="Path to a helm.toml file")
    parser.add_argument("--no-llm", action="store_true", help="Use the rule-based fallback only")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show tool calls on stderr")
    args = parser.parse_args(argv)
    logging.basicConfig(level=os.environ.get("HELM_LOG_LEVEL", "WARNING"), stream=sys.stderr)

    try:
        settings = load_settings(args.config)
    except ConfigError as exc:
        print(f"helm-ask: configuration problem: {exc}", file=sys.stderr)
        raise SystemExit(2) from None

    assistant = make_assistant(settings, use_llm=not args.no_llm)
    if assistant.mode == "fallback" and assistant.llm_problem and not args.no_llm:
        print(f"note: no LLM, using the rule-based fallback. Reason: {assistant.llm_problem}", file=sys.stderr)
    print(asyncio.run(_run(assistant, " ".join(args.question), args.verbose)))


if __name__ == "__main__":
    main()
