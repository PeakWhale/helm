"""The assistant: one entry point for the chat UI, the CLI and the evals.

It uses the LLM agent when a model is available, and the rule-based fallback
when not (or when the model fails). Both paths call the same tools and end with
the same "Sources" list.
"""

from __future__ import annotations

import logging
import re
from typing import Any, AsyncIterator

from .config import Settings
from .fallback import FallbackPlanner
from .sources import Event, Source, cited_ids, format_sources, unknown_citations
from .toolset import ToolSet, content_to_text

logger = logging.getLogger("helm.assistant")

MAX_SCHEMA_TABLES = 12
MAX_SCHEMA_CHARS = 6000


def clean_for_history(answer: str) -> str:
    """Remove the Sources list, notes and [S#] tags before reusing an answer as chat history."""
    text = answer.split("**Sources**")[0]
    text = re.sub(r"\[S\d+\]", "", text)
    text = "\n".join(line for line in text.splitlines() if not line.startswith("_"))
    return text.strip()


class Assistant:
    def __init__(
        self,
        toolset: ToolSet,
        settings: Settings,
        llm: Any | None = None,
        llm_label: str = "none",
        llm_problem: str | None = None,
    ):
        self.toolset = toolset
        self.settings = settings
        self.llm = llm
        self.llm_label = llm_label
        self.llm_problem = llm_problem
        self.fallback = FallbackPlanner(toolset.call, toolset.names, settings.saved_queries)
        self._schema: str | None = None
        self._graph: Any = None

    @property
    def mode(self) -> str:
        return "llm" if self.llm is not None else "fallback"

    async def answer(self, question: str, history: list[tuple[str, str]] | None = None) -> tuple[str, list[Source]]:
        """Run a question to the end and return (answer, sources)."""
        text, sources = "", []
        async for event in self.ask(question, history):
            if event.kind == "tool_result" and event.source is not None:
                sources.append(event.source)
            elif event.kind == "answer":
                text = event.text
        return text, sources

    async def ask(self, question: str, history: list[tuple[str, str]] | None = None) -> AsyncIterator[Event]:
        if self.llm is not None:
            try:
                async for event in self._ask_llm(question, history):
                    yield event
                return
            except Exception as exc:
                logger.warning("LLM run failed, using the fallback: %s", exc)
                yield Event("note", text=f"The LLM failed ({exc.__class__.__name__}: {exc}). Using the rule-based fallback.")

        result = await self.fallback.run(question)
        for source in result.sources:
            yield Event("tool_call", tool=source.tool, args=source.args)
            yield Event("tool_result", text=source.output, tool=source.tool, args=source.args, source=source)
        yield Event("answer", text=result.answer)

    # ------------------------------------------------------------------ LLM path

    async def schema_overview(self) -> str:
        if self._schema is not None:
            return self._schema
        try:
            listing = await self.toolset.call("list_tables", {})
        except Exception as exc:
            return f"(could not list tables: {exc})"
        parts = [listing]
        from .formatting import parse_markdown_table

        columns, rows = parse_markdown_table(listing)
        names = [r[0] for r in rows] if columns and columns[0] == "table" else []
        for name in names[:MAX_SCHEMA_TABLES]:
            try:
                described = await self.toolset.call("describe_table", {"table": name})
            except Exception as exc:
                described = f"Table: {name} (could not describe: {exc})"
            parts.append(described.split("\n\nSample rows:")[0])
        if len(names) > MAX_SCHEMA_TABLES:
            parts.append(f"({len(names) - MAX_SCHEMA_TABLES} more tables: call describe_table to see them)")
        text = "\n\n".join(parts)
        if len(text) > MAX_SCHEMA_CHARS:
            text = text[:MAX_SCHEMA_CHARS] + "\n(schema overview cut; call describe_table for more)"
        self._schema = text
        return text

    async def _graph_and_prompt(self) -> tuple[Any, str]:
        from .agent import SYSTEM_PROMPT, build_agent_graph

        schema = await self.schema_overview()
        dialect = "sqlite" if self.settings.db_kind == "sqlite" else "duckdb"
        prompt = SYSTEM_PROMPT.format(dialect=dialect, schema=schema)
        if self._graph is None:
            self._graph = build_agent_graph(self.llm, self.toolset, prompt, self.settings.llm.max_steps)
        return self._graph, prompt

    async def _ask_llm(self, question: str, history: list[tuple[str, str]] | None) -> AsyncIterator[Event]:
        from langchain_core.messages import HumanMessage

        from .agent import history_messages

        graph, _prompt = await self._graph_and_prompt()
        inputs = {"messages": [*history_messages(history), HumanMessage(content=question)], "sources": [], "rounds": 0}
        config = {"recursion_limit": 2 * self.settings.llm.max_steps + 6}
        final_text = ""
        sources: list[Source] = []
        async for update in graph.astream(inputs, config=config, stream_mode="updates"):
            for node, data in update.items():
                if not data:
                    continue
                if node == "agent":
                    message = data["messages"][-1]
                    calls = getattr(message, "tool_calls", None) or []
                    for call in calls:
                        yield Event("tool_call", tool=call["name"], args=dict(call.get("args") or {}))
                    if not calls:
                        final_text = content_to_text(message.content).strip()
                elif node == "tools":
                    for source in data.get("sources", []):
                        sources.append(source)
                        yield Event("tool_result", text=source.output, tool=source.tool, args=source.args, source=source)
        yield Event("answer", text=compose_answer(final_text, sources, self.llm_label))


def compose_answer(text: str, sources: list[Source], label: str) -> str:
    text = (text or "").strip() or "The model did not write an answer."
    notes = []
    if not sources:
        notes.append("_Note: no tool was used, so this answer is not backed by your data._")
    else:
        unknown = unknown_citations(text, sources)
        if unknown:
            notes.append(f"_Note: the answer cites {', '.join(unknown)}, which is not a real source._")
        if not cited_ids(text) - set(unknown):
            notes.append("_Note: the answer has no [S#] tags; check the sources below._")
    parts = [text]
    if sources:
        parts.append(format_sources(sources))
    parts.extend(notes)
    parts.append(f"_Answered by {label} with Helm tools._")
    return "\n\n".join(parts)
