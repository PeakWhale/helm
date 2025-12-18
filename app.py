import json
import chainlit as cl
from langchain_core.messages import HumanMessage

try:
    from src.graph import GraphRuntime
except ImportError:
    from graph import GraphRuntime


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


def _content_to_text(content) -> str:
    """
    MCP tool results can be returned as a list of content blocks, for example
    [{"type":"text","text":"..."}]
    Normalize to a single string for rendering.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        if content.get("type") == "text" and "text" in content:
            return str(content.get("text") or "")
        return _format_any(content)
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if item.get("type") == "text" and "text" in item:
                    parts.append(str(item.get("text") or ""))
                elif "text" in item:
                    parts.append(str(item.get("text") or ""))
                else:
                    parts.append(_format_any(item))
            else:
                parts.append(str(item))
        return "\n".join([p for p in parts if p]).strip()
    return str(content)


@cl.on_chat_start
async def start():
    runtime = await GraphRuntime.create()
    cl.user_session.set("runtime", runtime)

    await cl.Message(
        content=(
            "⚓ **PeakWhale Helm Online**\n\n"
            "MCP Host is running, and MCP tools are loaded from the local MCP tool server.\n\n"
            "Try:\n"
            "1) Quant: find the top Gambling High Risk customer\n"
            "2) Researcher: search loan PDFs for a customer id\n"
            "3) Market: get stock price for $AAPL or TSLA\n"
            "4) Sentiment: analyze sentiment on a sentence\n"
        )
    ).send()


@cl.on_chat_end
async def end():
    runtime = cl.user_session.get("runtime")
    if runtime:
        await runtime.aclose()


@cl.on_message
async def main(message: cl.Message):
    runtime: GraphRuntime = cl.user_session.get("runtime")
    graph = runtime.graph

    inputs = {"messages": [HumanMessage(content=message.content)]}

    final_answer = cl.Message(content="")
    await final_answer.send()

    try:
        async for output in graph.astream(inputs):
            for agent_name, agent_state in output.items():
                msgs = agent_state.get("messages") if isinstance(agent_state, dict) else None
                if not msgs:
                    continue

                last_msg = msgs[-1]

                tool_calls = _extract_tool_calls(last_msg)
                if tool_calls:
                    for call in tool_calls:
                        async with cl.Step(name=f"🔧 MCP tool call via {agent_name}") as step:
                            step.input = _format_any(call.get("args"))
                            step.output = f"Tool: {call.get('name')}"
                            await step.update()
                    continue

                if agent_name == "MCPTools":
                    tool_name = getattr(last_msg, "name", None) or "tool"
                    tool_call_id = getattr(last_msg, "tool_call_id", None)

                    async with cl.Step(name=f"📦 MCP tool result: {tool_name}") as step:
                        if tool_call_id:
                            step.input = f"tool_call_id: {tool_call_id}"
                        raw = getattr(last_msg, "content", "")
                        step.output = _format_any(_content_to_text(raw))
                        if _looks_like_error(step.output):
                            step.is_error = True
                        await step.update()
                    continue

                raw = getattr(last_msg, "content", "")
                content = _content_to_text(raw)

                if content.strip():
                    if final_answer.content:
                        final_answer.content += "\n" + content.strip()
                    else:
                        final_answer.content = content.strip()
                    await final_answer.update()

    except GeneratorExit:
        return
