"""The LLM planner: a small LangGraph tool-calling loop.

    question -> [agent] --tool calls--> [tools] --results--> [agent] -> ... -> answer

* The model sees the tool list and decides which tools to call, and how often.
* Every tool result is tagged [S1], [S2], ... and the model is told to cite the tags.
* The loop stops after ``max_steps`` tool rounds; the model must then answer
  from what it has.

Needs the agent extra: ``uv sync --extra agent``.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from .sources import Source, error_text
from .toolset import ToolSet

SYSTEM_PROMPT = """\
You are PeakWhale Helm, a careful research assistant for business data.
You can only state facts that come from tool results in this conversation.

How to work:
1. Choose the tools you need. You may call several tools, one after another.
2. For numbers and records, write ONE read-only SQL query (SELECT or WITH) and call `query`.
   Use the exact table and column names from the schema below. Prefer COUNT, SUM and GROUP BY
   over pulling many rows.
3. For facts written in documents (policies, contracts, applications, notes), call
   `search_documents`. Use `source_filter` to stay inside one file when you know an id or a file name.
4. If a tool returns an error, read it, fix the call and try again, or explain what is missing.
5. Every tool result starts with a tag like [S1]. Put the tag after each fact you use,
   for example: "Customer 2631d00b has 8 high risk transactions [S1]."
6. Never guess or invent numbers, ids, names or quotes. If the tools do not give the answer, say so.
7. Keep the answer short: the direct answer first, then the key details.

SQL dialect: {dialect}

Schema overview (for writing SQL only; facts must still come from tool results):
{schema}
"""

FORCE_ANSWER = (
    "You have used all your tool calls. Write the final answer now, using only the tool results "
    "above, with their [S#] tags. Say what is still missing."
)


class AgentState(TypedDict, total=False):
    messages: Annotated[list[Any], add_messages]
    sources: Annotated[list[Source], operator.add]
    rounds: int


def build_agent_graph(llm: Any, toolset: ToolSet, system_prompt: str, max_steps: int = 8):
    """Compile the agent graph. ``llm`` is any LangChain chat model with tool calling."""
    llm_with_tools = llm.bind_tools(toolset.langchain_tools())

    async def agent(state: AgentState) -> dict[str, Any]:
        messages = [SystemMessage(content=system_prompt), *state["messages"]]
        if state.get("rounds", 0) >= max_steps:
            response = await llm.ainvoke([*messages, HumanMessage(content=FORCE_ANSWER)])
            response = AIMessage(content=response.content)  # no more tool calls
        else:
            response = await llm_with_tools.ainvoke(messages)
        return {"messages": [response]}

    async def tools(state: AgentState) -> dict[str, Any]:
        last = state["messages"][-1]
        count = len(state.get("sources", []))
        messages, sources = [], []
        for call in getattr(last, "tool_calls", None) or []:
            count += 1
            source = Source(f"S{count}", call["name"], dict(call.get("args") or {}))
            try:
                source.output = await toolset.call(call["name"], source.args)
            except Exception as exc:  # the model sees the error and can try again
                source.output = error_text(exc)
                source.error = True
            sources.append(source)
            content = f"[{source.id}] " + (f"Error: {source.output}" if source.error else source.output)
            messages.append(ToolMessage(content=content, tool_call_id=call["id"], name=call["name"]))
        return {"messages": messages, "sources": sources, "rounds": state.get("rounds", 0) + 1}

    def route(state: AgentState) -> str:
        return "tools" if getattr(state["messages"][-1], "tool_calls", None) else END

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent)
    graph.add_node("tools", tools)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", route, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")
    return graph.compile()


def history_messages(history: list[tuple[str, str]] | None) -> list[Any]:
    """Turn (role, text) pairs from earlier turns into chat messages."""
    messages: list[Any] = []
    for role, text in history or []:
        if role == "user":
            messages.append(HumanMessage(content=text))
        else:
            messages.append(AIMessage(content=text))
    return messages
