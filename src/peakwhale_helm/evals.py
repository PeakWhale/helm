"""``helm-eval``: run the demo questions and print PASS / FAIL.

    uv run helm-eval --fresh-demo          # rule-based fallback, offline, fresh demo data
    uv run helm-eval --llm                 # the LLM agent (needs Ollama or an API)

A case passes when every expected text is in the answer (before the Sources list),
no rejected text is in it, and every expected tool was used.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import tempfile
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from .assistant import Assistant
from .config import ConfigError, Settings, load_settings
from .sources import Source

DEFAULT_CASES = Path("evals") / "demo_cases.json"


@dataclass
class CaseResult:
    id: str
    passed: bool
    problems: list[str]
    tools: list[str]
    seconds: float
    answer: str


def find_cases_file(path: str | Path | None = None) -> Path:
    if path:
        return Path(path)
    for candidate in (Path.cwd() / DEFAULT_CASES, Path(__file__).resolve().parents[2] / DEFAULT_CASES):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Could not find {DEFAULT_CASES}. Pass --cases PATH.")


def load_cases(path: str | Path | None = None) -> list[dict[str, Any]]:
    data = json.loads(find_cases_file(path).read_text(encoding="utf-8"))
    cases = data["cases"] if isinstance(data, dict) else data
    for case in cases:
        if not case.get("id") or not case.get("question"):
            raise ValueError(f"Each case needs an id and a question: {case}")
    return cases


def _contains(text: str, expected: str) -> bool:
    pattern = re.escape(expected.lower())
    if expected[:1].isalnum():
        pattern = r"(?<![0-9a-z])" + pattern
    if expected[-1:].isalnum():
        pattern = pattern + r"(?![0-9a-z])"
    return re.search(pattern, text.lower()) is not None


def check_case(case: dict[str, Any], answer: str, sources: list[Source]) -> list[str]:
    body = answer.split("**Sources**")[0]
    problems = []
    for item in case.get("expect", []):
        options = item if isinstance(item, list) else [item]
        if not any(_contains(body, str(option)) for option in options):
            problems.append("missing " + " or ".join(f'"{o}"' for o in options))
    for item in case.get("reject", []):
        if _contains(body, str(item)):
            problems.append(f'has "{item}"')
    used = {s.tool for s in sources}
    for tool in case.get("tools", []):
        if tool not in used:
            problems.append(f"tool {tool} not used")
    return problems


async def run_cases(assistant: Assistant, cases: list[dict[str, Any]]) -> list[CaseResult]:
    results = []
    for case in cases:
        start = time.perf_counter()
        try:
            answer, sources = await assistant.answer(case["question"])
            problems = check_case(case, answer, sources)
        except Exception as exc:  # one broken case should not stop the run
            answer, sources, problems = "", [], [f"crashed: {exc.__class__.__name__}: {exc}"]
        results.append(
            CaseResult(
                id=case["id"],
                passed=not problems,
                problems=problems,
                tools=[s.tool for s in sources],
                seconds=time.perf_counter() - start,
                answer=answer,
            )
        )
    return results


def print_report(results: list[CaseResult], mode: str, out=None) -> None:
    out = out or sys.stdout
    width = max((len(r.id) for r in results), default=10)
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        detail = ", ".join(r.tools) if r.passed else "; ".join(r.problems)
        print(f"{status}  {r.id:<{width}}  {r.seconds:5.2f}s  {detail}", file=out)
    passed = sum(r.passed for r in results)
    print(f"\n{passed}/{len(results)} passed (mode: {mode})", file=out)


def demo_settings(base: Settings, out_dir: Path) -> Settings:
    """Settings that point at fresh demo data in ``out_dir``, with the demo saved queries."""
    from .demo import DEMO_SAVED_QUERIES, generate

    paths = generate(out_dir)
    return replace(base.with_data(paths.ledger, paths.vault), saved_queries=DEMO_SAVED_QUERIES)


def main(argv: list[str] | None = None) -> None:
    from .cli import make_assistant

    parser = argparse.ArgumentParser(prog="helm-eval", description="Run the Helm demo evals")
    parser.add_argument("--cases", help=f"Eval file (default: {DEFAULT_CASES})")
    parser.add_argument("--config", help="Path to a helm.toml file")
    parser.add_argument("--llm", action="store_true", help="Use the configured LLM agent instead of the fallback")
    parser.add_argument("--fresh-demo", action="store_true", help="Build fresh demo data in a temp folder first")
    parser.add_argument("--show-answers", action="store_true", help="Print every answer")
    args = parser.parse_args(argv)

    try:
        settings = load_settings(args.config)
        cases = load_cases(args.cases)
    except (ConfigError, FileNotFoundError, ValueError) as exc:
        print(f"helm-eval: {exc}", file=sys.stderr)
        raise SystemExit(2) from None

    with tempfile.TemporaryDirectory(prefix="helm-eval-") as tmp:
        if args.fresh_demo:
            settings = demo_settings(settings, Path(tmp))
        assistant = make_assistant(settings, use_llm=args.llm)
        if args.llm and assistant.mode != "llm":
            print(f"helm-eval: no LLM available: {assistant.llm_problem}", file=sys.stderr)
            raise SystemExit(2)
        mode = f"llm {assistant.llm_label}" if assistant.mode == "llm" else "rule-based fallback"
        results = asyncio.run(run_cases(assistant, cases))
        assistant.toolset.toolbox.close()

    if args.show_answers:
        for r in results:
            print(f"--- {r.id}\n{r.answer}\n")
    print_report(results, mode)
    raise SystemExit(0 if all(r.passed for r in results) else 1)


if __name__ == "__main__":
    main()
