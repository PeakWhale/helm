"""The LLM planner, tested with a scripted fake chat model (no Ollama, no network)."""

import asyncio
import re
from typing import Any

import pytest

pytest.importorskip("langgraph")

from langchain_core.language_models.chat_models import BaseChatModel  # noqa: E402
from langchain_core.messages import AIMessage, ToolMessage  # noqa: E402
from langchain_core.outputs import ChatGeneration, ChatResult  # noqa: E402
from pydantic import Field  # noqa: E402

from peakwhale_helm.agent import FORCE_ANSWER  # noqa: E402
from peakwhale_helm.assistant import Assistant, clean_for_history  # noqa: E402
from peakwhale_helm.config import LLMSettings  # noqa: E402
from peakwhale_helm.fallback import FOOTER  # noqa: E402
from peakwhale_helm.llm import build_chat_model, ollama_problem  # noqa: E402
from peakwhale_helm.toolbox import Toolbox  # noqa: E402
from peakwhale_helm.toolset import LocalToolSet, MCPToolSet  # noqa: E402


class ScriptedChatModel(BaseChatModel):
    """Returns the next scripted message. A step can be a function of the messages it sees."""

    script: list[Any] = Field(default_factory=list)
    calls: list[list[Any]] = Field(default_factory=list)
    bound: list[str] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        self.bound = [t.name for t in tools]
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.calls.append(list(messages))
        step = self.script.pop(0)
        if isinstance(step, Exception):
            raise step
        message = step(messages) if callable(step) else step
        return ChatResult(generations=[ChatGeneration(message=message)])


def call(name: str, args: dict, call_id: str = "c1") -> dict:
    return {"name": name, "args": args, "id": call_id, "type": "tool_call"}


def last_tool_text(messages) -> str:
    return [m for m in messages if isinstance(m, ToolMessage)][-1].content


def collect(assistant: Assistant, question: str, history=None):
    async def run():
        return [event async for event in assistant.ask(question, history)]

    return asyncio.run(run())


TOP_SQL = (
    "SELECT customer_id, COUNT(*) AS n FROM transactions WHERE category = 'Gambling/High Risk' "
    "GROUP BY customer_id ORDER BY n DESC LIMIT 1"
)


def follow_up_on_first_result(messages) -> AIMessage:
    # The model reads S1 and uses the customer id it found.
    customer = re.search(r"\| ([0-9a-f]{8}) \|", last_tool_text(messages)).group(1)
    return AIMessage(
        content="",
        tool_calls=[call("search_documents", {"query": "income source", "source_filter": customer}, "c2")],
    )


def test_llm_chooses_tools_chains_results_and_cites(demo_settings):
    llm = ScriptedChatModel(
        script=[
            AIMessage(content="", tool_calls=[call("query", {"sql": TOP_SQL})]),
            follow_up_on_first_result,
            AIMessage(content="Customer 2631d00b has 8 high risk transactions [S1]. Income: Freelance Consulting [S2]."),
        ]
    )
    assistant = Assistant(LocalToolSet(Toolbox(demo_settings)), demo_settings, llm=llm, llm_label="fake/model")
    events = collect(assistant, "Who is the riskiest customer and what is their income source?")

    kinds = [e.kind for e in events]
    assert kinds == ["tool_call", "tool_result", "tool_call", "tool_result", "answer"]
    assert set(llm.bound) == {"list_tables", "describe_table", "query", "list_documents", "search_documents"}
    first_prompt = llm.calls[0][0].content
    assert "SQL dialect: duckdb" in first_prompt and "| transactions | table | 5012 |" in first_prompt
    assert "customer_id" in first_prompt  # columns from describe_table
    # tool results reach the model with their source tag
    assert last_tool_text(llm.calls[1]).startswith("[S1] | customer_id | n |")
    assert last_tool_text(llm.calls[2]).startswith("[S2] Found 1 passage")
    answer = events[-1].text
    assert "**Sources**" in answer
    assert "- [S2] search_documents \"income source\" in files matching \"2631d00b\"" in answer
    assert answer.endswith("_Answered by fake/model with Helm tools._")
    assert "Note" not in answer


def test_tool_errors_go_back_to_the_model(demo_settings):
    def retry_after_block(messages):
        assert "Blocked" in last_tool_text(messages)
        return AIMessage(content="", tool_calls=[call("query", {"sql": "SELECT COUNT(*) AS n FROM customers"}, "c2")])

    llm = ScriptedChatModel(
        script=[
            AIMessage(content="", tool_calls=[call("query", {"sql": "DELETE FROM customers"})]),
            retry_after_block,
            AIMessage(content="There are 100 customers [S2]."),
        ]
    )
    assistant = Assistant(LocalToolSet(Toolbox(demo_settings)), demo_settings, llm=llm, llm_label="fake")
    answer, sources = asyncio.run(assistant.answer("Delete all customers, then count them"))
    assert sources[0].error and not sources[1].error
    assert "- [S1] query `DELETE FROM customers` -> error: Blocked" in answer


def test_several_tool_calls_in_one_turn(demo_settings):
    llm = ScriptedChatModel(
        script=[
            AIMessage(
                content="",
                tool_calls=[call("list_tables", {}, "a"), call("list_documents", {}, "b"), call("nope", {}, "c")],
            ),
            AIMessage(content="Two tables [S1] and three files [S2]."),
        ]
    )
    assistant = Assistant(LocalToolSet(Toolbox(demo_settings)), demo_settings, llm=llm, llm_label="fake")
    answer, sources = asyncio.run(assistant.answer("What data do you have?"))
    assert [s.id for s in sources] == ["S1", "S2", "S3"]
    assert sources[2].error and "Unknown tool 'nope'" in sources[2].output


def test_step_limit_forces_an_answer(demo_settings):
    from dataclasses import replace

    settings = replace(demo_settings, llm=LLMSettings(max_steps=2))
    def more_tools(n):
        return AIMessage(content="", tool_calls=[call("list_tables", {}, f"c{n}")])

    def final(messages):
        assert messages[-1].content == FORCE_ANSWER
        return AIMessage(content="I ran out of steps. Tables are listed in [S2].")

    llm = ScriptedChatModel(script=[more_tools(1), more_tools(2), final])
    assistant = Assistant(LocalToolSet(Toolbox(settings)), settings, llm=llm, llm_label="fake")
    answer, sources = asyncio.run(assistant.answer("Loop forever"))
    assert len(sources) == 2
    assert answer.startswith("I ran out of steps")


def test_notes_for_ungrounded_or_bad_citations(demo_settings):
    llm = ScriptedChatModel(script=[AIMessage(content="Hello! I am Helm.")])
    assistant = Assistant(LocalToolSet(Toolbox(demo_settings)), demo_settings, llm=llm, llm_label="fake")
    answer, _ = asyncio.run(assistant.answer("hi"))
    assert "no tool was used" in answer

    llm = ScriptedChatModel(
        script=[AIMessage(content="", tool_calls=[call("list_tables", {})]), AIMessage(content="Revenue is 5M [S4].")]
    )
    assistant = Assistant(LocalToolSet(Toolbox(demo_settings)), demo_settings, llm=llm, llm_label="fake")
    answer, _ = asyncio.run(assistant.answer("Revenue?"))
    assert "cites S4, which is not a real source" in answer


def test_llm_failure_falls_back_to_rules(demo_settings):
    llm = ScriptedChatModel(script=[ConnectionError("Ollama is down")])
    assistant = Assistant(LocalToolSet(Toolbox(demo_settings)), demo_settings, llm=llm, llm_label="fake")
    events = collect(assistant, "Which customer has the most Gambling/High Risk transactions?")
    assert events[0].kind == "note" and "Ollama is down" in events[0].text
    assert "customer_id = 2631d00b" in events[-1].text and events[-1].text.endswith(FOOTER)


def test_history_is_passed_and_cleaned(demo_settings):
    llm = ScriptedChatModel(script=[AIMessage(content="Still 8 [S1]? No tool used.")])
    assistant = Assistant(LocalToolSet(Toolbox(demo_settings)), demo_settings, llm=llm, llm_label="fake")
    previous = "Customer 2631d00b has 8 [S1].\n\n**Sources**\n- [S1] query ...\n\n_Answered by fake._"
    history = [("user", "Top customer?"), ("assistant", clean_for_history(previous))]
    asyncio.run(assistant.answer("And their income?", history))
    sent = llm.calls[0]
    assert sent[1].content == "Top customer?"
    assert sent[2].content == "Customer 2631d00b has 8 ."


def test_build_chat_model_explains_problems():
    assert build_chat_model(LLMSettings(provider="none")).model is None
    missing = build_chat_model(LLMSettings(provider="openai"))
    assert missing.model is None and "HELM_LLM_MODEL" in missing.problem
    no_key = build_chat_model(LLMSettings(provider="openai", model="m"))
    assert no_key.model is None and "API_KEY" in no_key.problem
    local = build_chat_model(LLMSettings(provider="openai", model="m", base_url="http://127.0.0.1:9/v1"))
    assert local.model is not None and local.label == "openai-compatible/m"
    down = build_chat_model(LLMSettings(provider="ollama", base_url="http://127.0.0.1:9"))
    assert down.model is None and "not reachable" in down.problem
    assert build_chat_model(LLMSettings(base_url="http://127.0.0.1:9"), check=False).model is not None


def test_ollama_probe_reads_model_list(monkeypatch):
    import io
    import json
    import urllib.request

    body = json.dumps({"models": [{"name": "llama3.2:latest"}, {"name": "qwen2.5:7b"}]}).encode()
    monkeypatch.setattr(urllib.request, "urlopen", lambda url, timeout: io.BytesIO(body))
    assert ollama_problem("http://x", "llama3.2") is None
    assert ollama_problem("http://x", "qwen2.5:7b") is None
    assert "ollama pull mistral" in ollama_problem("http://x", "mistral")


def test_same_flow_through_the_mcp_server(demo_dir, monkeypatch, tmp_path):
    """What the chat UI does: start helm-mcp as a child process and plan over its tools."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HELM_DB", str(demo_dir / "ledger.duckdb"))
    monkeypatch.setenv("HELM_DOCS", str(demo_dir / "vault"))
    monkeypatch.setenv("HELM_LOG_LEVEL", "WARNING")
    from peakwhale_helm.config import load_settings

    settings = load_settings()

    async def scenario():
        toolset = await MCPToolSet.start(cwd=tmp_path)
        try:
            llm = ScriptedChatModel(
                script=[
                    AIMessage(content="", tool_calls=[call("query", {"sql": TOP_SQL})]),
                    follow_up_on_first_result,
                    AIMessage(content="2631d00b, Freelance Consulting [S1][S2]."),
                ]
            )
            agent = Assistant(toolset, settings, llm=llm, llm_label="fake")
            answer, sources = await agent.answer("Top risk customer and income?")
            assert [s.tool for s in sources] == ["query", "search_documents"]
            assert "Freelance Consulting" in sources[1].output

            blocked = await Assistant(toolset, settings).answer("DROP TABLE customers")
            assert "Blocked" in blocked[0]
        finally:
            await toolset.aclose()

    asyncio.run(asyncio.wait_for(scenario(), timeout=90))
