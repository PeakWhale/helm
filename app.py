"""PeakWhale Helm chat UI (Chainlit).

Run:  uv sync --extra ui  &&  uv run chainlit run app.py

The UI is just one MCP client: it starts the Helm MCP server (helm-mcp) as a
child process and plans over its tools with the LLM agent, or with the
rule-based fallback when no LLM is available.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import chainlit as cl

from peakwhale_helm.assistant import Assistant, clean_for_history
from peakwhale_helm.config import load_settings
from peakwhale_helm.demo import DEMO_SAVED_QUERIES
from peakwhale_helm.llm import build_chat_model
from peakwhale_helm.toolset import MCPToolSet

HISTORY_TURNS = 3
MAX_STEP_OUTPUT = 4000

DEMO_QUESTIONS = [
    "Which customer has the most Gambling/High Risk transactions?",
    "Find the top high risk customer and check their loan application for income source.",
    "What does the credit policy say about gambling transactions?",
    "How many customers are in the ledger?",
]
GENERIC_QUESTIONS = [
    "What tables are in the database?",
    "Describe the <table> table",
    "What documents do you have?",
    "What does the <document> say about <topic>?",
]


@cl.on_chat_start
async def start() -> None:
    settings = load_settings()
    toolset = await MCPToolSet.start()
    chat = await asyncio.to_thread(build_chat_model, settings.llm)
    assistant = Assistant(toolset, settings, llm=chat.model, llm_label=chat.label, llm_problem=chat.problem)
    cl.user_session.set("assistant", assistant)
    cl.user_session.set("history", [])

    if chat.model is not None:
        planner = f"LLM planner: **{chat.label}**"
    else:
        planner = f"LLM planner: off, using the rule-based fallback. Reason: {chat.problem}"
    docs = settings.docs_dir.name if settings.docs_dir else "none"
    questions = DEMO_QUESTIONS if settings.saved_queries == DEMO_SAVED_QUERIES else GENERIC_QUESTIONS
    examples = "\n".join(f"{n}. {q}" for n, q in enumerate(questions, start=1))
    await cl.Message(
        content=(
            "⚓ **PeakWhale Helm is ready**\n\n"
            f"- Database: {settings.db_kind} ({Path(settings.db_target).name})\n"
            f"- Documents: {docs} ({settings.retriever} search)\n"
            f"- Tools: {', '.join(toolset.names)}\n"
            f"- {planner}\n\n"
            f"Try:\n{examples}"
        )
    ).send()


@cl.on_message
async def on_message(message: cl.Message) -> None:
    assistant: Assistant | None = cl.user_session.get("assistant")
    if assistant is None:
        await cl.Message(content="Helm is still starting. Please try again in a moment.").send()
        return
    history = cl.user_session.get("history") or []

    open_steps: list[cl.Step] = []
    answer = ""
    async for event in assistant.ask(message.content, history):
        if event.kind == "tool_call":
            step = cl.Step(name=event.tool or "tool", type="tool")
            step.input = event.args or {}
            await step.send()
            open_steps.append(step)
        elif event.kind == "tool_result":
            step = open_steps.pop(0) if open_steps else cl.Step(name=event.tool or "tool", type="tool")
            if event.source is not None:
                step.name = f"{event.tool} [{event.source.id}]"
                step.is_error = event.source.error
            text = event.text or ""
            step.output = text if len(text) <= MAX_STEP_OUTPUT else text[:MAX_STEP_OUTPUT] + "\n\n[cut]"
            await step.update()
        elif event.kind == "note":
            await cl.Message(content=f"_{event.text}_").send()
        elif event.kind == "answer":
            answer = event.text

    await cl.Message(content=answer or "No answer.").send()
    history = [*history, ("user", message.content), ("assistant", clean_for_history(answer))]
    cl.user_session.set("history", history[-2 * HISTORY_TURNS :])


@cl.on_chat_end
async def end() -> None:
    assistant: Assistant | None = cl.user_session.get("assistant")
    if assistant is not None:
        await assistant.toolset.aclose()
