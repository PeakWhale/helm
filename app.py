import json
import chainlit as cl
from langchain_core.messages import HumanMessage
from src.graph import app as graph_app


def _extract_tool_calls(msg):
    """
    Normalize tool calls across providers.
    Returns a list of dicts: {id, name, args}
    """
    calls = []

    direct = getattr(msg, "tool_calls", None)
    if direct:
        calls = direct
    else:
        ak = getattr(msg, "additional_kwargs", {}) or {}
        calls = ak.get("tool_calls") or []

    normalized = []
    for c in calls:
        name = c.get("name") or (c.get("function", {}) or {}).get("name")

        args = (
            c.get("args")
            or c.get("arguments")
            or (c.get("function", {}) or {}).get("arguments")
        )

        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                pass

        call_id = c.get("id") or c.get("tool_call_id")

        normalized.append(
            {
                "id": call_id,
                "name": name or "unknown_tool",
                "args": args or {},
            }
        )

    return normalized


def _format_any(value, max_len=2500):
    try:
        if isinstance(value, (dict, list)):
            text = json.dumps(value, indent=2, ensure_ascii=False)
        else:
            text = str(value)
    except Exception:
        text = repr(value)

    if len(text) > max_len:
        return text[:max_len] + "\n\n[truncated]"
    return text


def _looks_like_error(text):
    t = (text or "").lower()
    return ("error" in t) or ("exception" in t) or ("traceback" in t) or ("failed" in t)


@cl.on_chat_start
async def start():
    cl.user_session.set("graph", graph_app)
    await cl.Message(
        content=(
            "⚓ **PeakWhale Helm Online.**\n\n"
            "Try:\n"
            "1) Quant: find the top Gambling/High Risk customer\n"
            "2) Researcher: search loan PDFs for a customer id\n"
            "3) Market: get stock price for $AAPL or TSLA\n"
            "4) Sentiment: analyze sentiment on a sentence\n"
        )
    ).send()


@cl.on_message
async def main(message: cl.Message):
    graph = cl.user_session.get("graph")
    inputs = {"messages": [HumanMessage(content=message.content)]}

    final_answer = cl.Message(content="")
    await final_answer.send()

    async for output in graph.astream(inputs):
        for agent_name, agent_state in output.items():
            msgs = agent_state.get("messages") if isinstance(agent_state, dict) else None
            if not msgs:
                continue

            last_msg = msgs[-1]

            tool_calls = _extract_tool_calls(last_msg)
            if tool_calls:
                for call in tool_calls:
                    async with cl.Step(name=f"🤔 {agent_name} tool call") as step:
                        step.input = _format_any(call.get("args"))
                        step.output = f"Tool: {call.get('name')}"
                        await step.update()
                continue

            if agent_name == "Tools":
                tool_name = getattr(last_msg, "name", None) or "tool"
                tool_call_id = getattr(last_msg, "tool_call_id", None)

                async with cl.Step(name=f"💾 {tool_name} result") as step:
                    if tool_call_id:
                        step.input = f"tool_call_id: {tool_call_id}"
                    step.output = _format_any(getattr(last_msg, "content", ""))
                    if _looks_like_error(step.output):
                        step.is_error = True
                    await step.update()
                continue

            content = getattr(last_msg, "content", "") or ""
            if content.strip():
                if final_answer.content:
                    final_answer.content += "\n" + content.strip()
                else:
                    final_answer.content = content.strip()
                await final_answer.update()
