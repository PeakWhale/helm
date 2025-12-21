import operator
import re
import uuid
import os
import sys
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, TypedDict, List, Optional, Any, Callable

from datetime import datetime


logger = logging.getLogger("helm.graph")
logging.basicConfig(
    level=os.environ.get("HELM_LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)


def _truthy_env(name: str, default: str = "1") -> bool:
    v = os.environ.get(name, default)
    return str(v).strip().lower() not in {"0", "false", "no", "off", ""}


def _build_ollama_llm() -> tuple[Any, Optional[str]]:
    """
    Returns (llm, model_name). If Ollama is disabled or not available, returns (None, None).
    """
    if not _truthy_env("HELM_USE_OLLAMA", "1"):
        return None, None

    model_name = os.environ.get("HELM_OLLAMA_MODEL", "llama3.2")
    base_url = os.environ.get("HELM_OLLAMA_BASE_URL", "").strip() or None

    try:
        try:
            from langchain_ollama import ChatOllama
        except Exception:
            from langchain_community.chat_models import ChatOllama  # fallback

        kwargs = {
            "model": model_name,
            "temperature": float(os.environ.get("HELM_OLLAMA_TEMPERATURE", "0.2")),
        }
        if base_url:
            kwargs["base_url"] = base_url

        llm = ChatOllama(**kwargs)
        logger.info("Ollama enabled. Model: %s", model_name)
        return llm, model_name

    except Exception as e:
        logger.warning("Ollama unavailable. Falling back to template answers. Details: %s", e)
        return None, None


class AgentState(TypedDict, total=False):
    messages: Annotated[Any, operator.add]
    next: str

    last_user_text: Optional[str]
    mode: Optional[str]

    customer_id: Optional[str]
    high_risk_count: Optional[int]

    doc_query: Optional[str]
    loan_excerpt: Optional[str]
    income_source: Optional[str]
    doc_fact_line: Optional[str]

    ticker: Optional[str]
    market_result: Optional[str]
    sentiment_result: Optional[str]

    quant_attempts: int
    research_attempts: int
    market_attempts: int
    sentiment_attempts: int


TOP_HIGH_RISK_SQL = (
    "SELECT customer_id, COUNT(*) AS high_risk_count "
    "FROM transactions "
    "WHERE category = 'Gambling/High Risk' "
    "GROUP BY customer_id "
    "ORDER BY 2 DESC "
    "LIMIT 1"
)


def _get_last_user_text(state: AgentState) -> str:
    msgs = state.get("messages", [])
    for m in reversed(msgs):
        if getattr(m, "type", None) == "human":
            return (m.content or "")
    return ""


def _extract_customer_id_from_text(user_text: str) -> Optional[str]:
    m = re.search(r"\b([a-f0-9]{8})\b", (user_text or "").lower())
    return m.group(1) if m else None


def _extract_ticker(user_text: str) -> Optional[str]:
    t = (user_text or "").strip()

    stop = {
        "GET", "LATEST", "STOCK", "PRICE", "TICKER", "WHAT", "IS", "THE", "OF", "FOR",
        "PLEASE", "CAN", "YOU", "QUOTE", "MARKET", "DATA", "CUSTOMER", "RESEARCHER",
        "QUANT", "SENTIMENT", "ANALYZE", "INCOME", "SOURCE", "LOAN", "APPLICATION",
        "CHECK", "FIND", "THEIR", "RESILIENCE", "STRONG", "SECTOR", "TECH", "SHOWING"
    }

    # 1. Explicit $TICKER
    # We find ALL matches and pick the first one that looks valid? Usually $TICKER is unambiguous.
    m = re.search(r"\$([A-Za-z][A-Za-z0-9\.\-]{0,9})\b", t)
    if m:
        return m.group(1).upper()

    # 2. "for X" or "of X", but X must not be a stopword
    # usage: re.finditer to check all "for X" candidates
    for pat in [r"(?i)\bfor\s+\$?([A-Za-z][A-Za-z0-9\.\-]{0,9})\b", r"(?i)\bof\s+\$?([A-Za-z][A-Za-z0-9\.\-]{0,9})\b"]:
        for match in re.finditer(pat, t):
            candidate = match.group(1).upper()
            if candidate not in stop:
                return candidate

    # 3. Last fallback: just the last valid capitalized token
    tokens = re.findall(r"\b[A-Za-z][A-Za-z0-9\.\-]{0,9}\b", t)
    candidates = [tok.upper() for tok in tokens if tok.upper() not in stop]
    return candidates[-1] if candidates else None


def _detect_mode(user_text: str) -> str:
    t = (user_text or "").lower()

    wants_top = ("most" in t or "top" in t) and ("high risk" in t or "gambling" in t)
    wants_loan = any(
        k in t
        for k in [
            "loan application",
            "loan app",
            "pdf",
            "income source",
            "source of income",
            "annual income",
            "speculative trading",
            "do not engage",
        ]
    )

    wants_stock = bool(re.search(r"\$[A-Za-z]{1,10}\b", user_text)) or bool(
        re.search(r"\b(stock|share)\s+price\b", t)
        or re.search(r"\bprice\s+of\b", t)
        or re.search(r"\bticker\b", t)
        or re.search(r"\bquote\b", t)
        or re.search(r"\blatest\s+price\b", t)
    )

    wants_sentiment = ("analyze sentiment" in t) or t.strip().startswith("sentiment:")

    if wants_top and wants_loan and wants_stock and wants_sentiment:
        return "omni_mode"

    if wants_top and wants_loan:
        return "top_then_loan"

    if wants_sentiment:
        return "sentiment"
    if wants_stock:
        return "stock_price"

    if wants_top:
        return "top_risk"
    if wants_loan:
        return "loan_query"

    return "unknown"


def _extract_doc_query(user_text: str) -> Optional[str]:
    seg = (user_text or "").strip()
    seg_l = seg.lower()

    if "income source" in seg_l or "source of income" in seg_l or "stated income source" in seg_l:
        return "income source"

    m = re.search(r"(?i)\bthen\b.*?\bfor\b\s+(.+)$", seg)
    if m:
        candidate = m.group(1).strip().strip(".")
        if candidate:
            candidate_l = candidate.lower()
            if "income source" in candidate_l or "source of income" in candidate_l:
                return "income source"
            return candidate

    quoted = re.findall(r"['\"]([^'\"]+)['\"]", seg)
    for q in quoted:
        ql = q.lower()
        if "gambling" in ql or "high risk" in ql:
            continue
        return q.strip()

    cleaned = re.sub(r"(?i)^\s*(confirm|check|pull|search|find|retrieve|tell me)\b\s*", "", seg).strip()
    cleaned = cleaned.strip().strip(".").strip()

    return cleaned if len(cleaned) >= 3 else None


def _extract_customer_id_and_count(table_text: str):
    text = table_text or ""
    lines = [ln for ln in text.splitlines() if ln.strip()]

    pipe_lines = [ln for ln in lines if ln.strip().startswith("|")]
    if len(pipe_lines) >= 3:
        data_line = pipe_lines[2].strip().strip("|")
        parts = [p.strip() for p in data_line.split("|") if p.strip()]
        if len(parts) >= 2:
            cid = parts[0].lower()
            cnt_str = re.sub(r"[^\d]", "", parts[1])
            cnt = int(cnt_str) if cnt_str.isdigit() else None
            return cid, cnt

    m = re.search(r"\b([a-f0-9]{8})\b\s+(\d+)\b", text.lower())
    if m:
        return m.group(1), int(m.group(2))

    return None, None


def _extract_income_source(text: str) -> Optional[str]:
    if not text:
        return None

    patterns = [
        r"(?i)\bstated\s+income\s+source\b\s*[:\-]\s*([^\n\r]+)",
        r"(?i)\bincome\s*source\b\s*[:\-]\s*([^\n\r]+)",
        r"(?i)\bsource\s*of\s*income\b\s*[:\-]\s*([^\n\r]+)",
    ]
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            return m.group(1).strip()

    return None


def _extract_fact_line(text: str, doc_query: Optional[str]) -> Optional[str]:
    if not text or not doc_query:
        return None

    q = doc_query.lower()

    for ln in text.splitlines():
        lnl = ln.lower().strip()
        if not lnl:
            continue

        if "speculative" in q:
            if "do not engage" in lnl or "speculative" in lnl:
                return ln.strip()

        if "annual income" in q:
            if lnl.startswith("annual income"):
                return ln.strip()

        if "income source" in q:
            if "stated income source" in lnl or lnl.startswith("income source"):
                return ln.strip()

    return None


def _tool_content_to_text(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        if content.get("type") == "text" and "text" in content:
            return str(content.get("text") or "")
        return json.dumps(content, ensure_ascii=False, indent=2)
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
                    parts.append(json.dumps(item, ensure_ascii=False))
            else:
                parts.append(str(item))
        return "\n".join([p for p in parts if p]).strip()
    return str(content)


def supervisor_node(state: AgentState):
    user_text = _get_last_user_text(state)
    mode = _detect_mode(user_text)

    is_new_request = user_text != (state.get("last_user_text") or "")
    updates: AgentState = {"mode": mode, "last_user_text": user_text}

    if is_new_request:
        updates["quant_attempts"] = 0
        updates["research_attempts"] = 0
        updates["market_attempts"] = 0
        updates["sentiment_attempts"] = 0

        updates["doc_query"] = None
        updates["loan_excerpt"] = None
        updates["income_source"] = None
        updates["doc_fact_line"] = None

        updates["ticker"] = None
        updates["market_result"] = None
        updates["sentiment_result"] = None

        if mode in ["top_risk", "top_then_loan", "omni_mode"]:
            updates["customer_id"] = None
            updates["high_risk_count"] = None

    explicit_cid = _extract_customer_id_from_text(user_text)
    if explicit_cid:
        updates["customer_id"] = explicit_cid

    if mode in ["loan_query", "top_then_loan", "omni_mode"]:
        doc_query = _extract_doc_query(user_text)
        if doc_query:
            updates["doc_query"] = doc_query

    if mode == "stock_price" or mode == "omni_mode":
        ticker = _extract_ticker(user_text)
        if ticker:
            updates["ticker"] = ticker

    customer_id = updates.get("customer_id", state.get("customer_id"))
    has_customer = bool(customer_id)

    has_count = (
        updates.get("high_risk_count") if "high_risk_count" in updates else state.get("high_risk_count")
    ) is not None

    has_doc = bool(updates.get("loan_excerpt") or state.get("loan_excerpt"))
    has_income = bool(updates.get("income_source") or state.get("income_source"))

    has_market = bool(updates.get("market_result") or state.get("market_result"))
    has_sentiment = bool(updates.get("sentiment_result") or state.get("sentiment_result"))

    quant_attempts = updates.get("quant_attempts", state.get("quant_attempts", 0))
    research_attempts = updates.get("research_attempts", state.get("research_attempts", 0))
    market_attempts = updates.get("market_attempts", state.get("market_attempts", 0))
    sentiment_attempts = updates.get("sentiment_attempts", state.get("sentiment_attempts", 0))

    if mode == "stock_price":
        if has_market or market_attempts >= 1:
            return {"next": "Final", **updates}
        return {"next": "Market", **updates}

    if mode == "sentiment":
        if has_sentiment or sentiment_attempts >= 1:
            return {"next": "Final", **updates}
        return {"next": "Sentiment", **updates}

    if mode == "top_risk":
        if quant_attempts >= 2:
            return {"next": "Final", **updates}
        if has_customer and has_count:
            return {"next": "Final", **updates}
        return {"next": "Quant", **updates}

    if mode == "loan_query":
        if not has_customer:
            if quant_attempts >= 2:
                return {"next": "Final", **updates}
            return {"next": "Quant", **updates}

        if has_income or has_doc:
            return {"next": "Final", **updates}

        if research_attempts >= 2:
            return {"next": "Final", **updates}
        return {"next": "Researcher", **updates}

    if mode == "omni_mode":
        # 1. Quant
        if not (has_customer and has_count):
            if quant_attempts >= 2:
                # If quant fails, we might still want other parts, but let's assume chain depends on it
                pass 
            else:
                return {"next": "Quant", **updates}

        # 2. Researcher
        if not (has_income or has_doc):
            if research_attempts >= 2:
                pass
            else:
                return {"next": "Researcher", **updates}

        # 3. Market
        if not (has_market or market_attempts >= 1):
             return {"next": "Market", **updates}

        # 4. Sentiment
        if not (has_sentiment or sentiment_attempts >= 1):
             return {"next": "Sentiment", **updates}

        return {"next": "Final", **updates}

    return {"next": "Final", **updates}


def quant_node(state: AgentState):
    from langchain_core.messages import AIMessage
    tool_call_id = str(uuid.uuid4())
    msg = AIMessage(
        content="",
        tool_calls=[
            {
                "id": tool_call_id,
                "name": "query_ledger",
                "args": {"sql_query": TOP_HIGH_RISK_SQL},
            }
        ],
    )
    return {"messages": [msg], "quant_attempts": state.get("quant_attempts", 0) + 1}


def researcher_node(state: AgentState):
    from langchain_core.messages import AIMessage
    customer_id = state.get("customer_id") or ""
    doc_query = state.get("doc_query") or "income source"

    tool_call_id = str(uuid.uuid4())
    msg = AIMessage(
        content="",
        tool_calls=[
            {
                "id": tool_call_id,
                "name": "search_loan_documents",
                "args": {"customer_id": customer_id, "query": doc_query},
            }
        ],
    )
    return {"messages": [msg], "research_attempts": state.get("research_attempts", 0) + 1


}


def market_node(state: AgentState):
    from langchain_core.messages import AIMessage
    ticker = state.get("ticker") or ""
    tool_call_id = str(uuid.uuid4())
    msg = AIMessage(
        content="",
        tool_calls=[
            {
                "id": tool_call_id,
                "name": "get_stock_price",
                "args": {"ticker": ticker},
            }
        ],
    )
    return {"messages": [msg], "market_attempts": state.get("market_attempts", 0) + 1}


def sentiment_node(state: AgentState):
    from langchain_core.messages import AIMessage
    user_text = _get_last_user_text(state)
    tool_call_id = str(uuid.uuid4())
    msg = AIMessage(
        content="",
        tool_calls=[
            {
                "id": tool_call_id,
                "name": "analyze_sentiment",
                "args": {"text": user_text},
            }
        ],
    )
    return {"messages": [msg], "sentiment_attempts": state.get("sentiment_attempts", 0) + 1}


def extractor_node(state: AgentState):
    if not state.get("messages"):
        return {}

    last_msg = state["messages"][-1]
    if getattr(last_msg, "type", "") != "tool":
        return {}

    tool_name = getattr(last_msg, "name", "") or ""
    raw_content = getattr(last_msg, "content", "")
    content = _tool_content_to_text(raw_content)

    updates: AgentState = {}

    if tool_name == "query_ledger":
        cid, cnt = _extract_customer_id_and_count(content)
        if cid:
            updates["customer_id"] = cid
        if cnt is not None:
            updates["high_risk_count"] = cnt

    if tool_name == "search_loan_documents":
        if content.startswith("Document Excerpt for"):
            updates["loan_excerpt"] = content

            inc = _extract_income_source(content)
            if inc:
                updates["income_source"] = inc

            fact = _extract_fact_line(content, state.get("doc_query"))
            if fact:
                updates["doc_fact_line"] = fact

    if tool_name == "get_stock_price":
        updates["market_result"] = content

    if tool_name == "analyze_sentiment":
        updates["sentiment_result"] = content

    return updates


def final_template_node(state: AgentState):
    from langchain_core.messages import AIMessage
    mode = state.get("mode") or "unknown"

    customer_id = state.get("customer_id")
    cnt = state.get("high_risk_count")

    doc_query = state.get("doc_query") or ""
    income_source = state.get("income_source")
    excerpt = state.get("loan_excerpt")
    fact_line = state.get("doc_fact_line")

    ticker = state.get("ticker")
    market_result = state.get("market_result")
    sentiment_result = state.get("sentiment_result")

    if mode == "stock_price":
        if not ticker:
            return {"messages": [AIMessage(content="Tell me the ticker, like $AAPL or TSLA, and I will fetch the latest price.")]}
        if market_result:
            return {"messages": [AIMessage(content=market_result)]}
        return {"messages": [AIMessage(content=f"Market data is unavailable right now for {ticker}. Try again shortly.")]}

    if mode == "sentiment":
        if sentiment_result:
            return {"messages": [AIMessage(content=sentiment_result)]}
        return {"messages": [AIMessage(content="Sentiment analysis is unavailable right now.")]}

    if mode == "top_risk":
        if not customer_id:
            return {"messages": [AIMessage(content="I could not find a customer for the Gambling or High Risk category.")]}
        line = f"Customer with the most 'Gambling/High Risk' transactions: **{customer_id}**"
        if cnt is not None:
            line += f" (**{cnt}** transactions)."
        else:
            line += "."
        return {"messages": [AIMessage(content=line)]}
    
    if mode == "final_answer":
        return {"messages": [AIMessage(content=f"Final Answer: {state.get('last_user_text')}") ]}

    if mode in ["loan_query", "top_then_loan"]:
        if not customer_id:
            return {"messages": [AIMessage(content="I could not determine a customer id to look up the loan application.")]}
        lead = f"Customer: **{customer_id}**"
        if cnt is not None and mode == "top_then_loan":
            lead += f" (High Risk count: **{cnt}**)."

        if "income source" in doc_query.lower() and income_source:
            return {"messages": [AIMessage(content=f"{lead}\nIncome source stated on their loan application: **{income_source}**")]}

        if fact_line:
            return {"messages": [AIMessage(content=f"{lead}\nRelevant line from the loan application:\n{fact_line}")]}

        if excerpt:
            return {"messages": [AIMessage(content=f"{lead}\nMost relevant excerpt:\n\n{excerpt}")]}

        return {"messages": [AIMessage(content=f"{lead}\nI could not retrieve the loan application PDF for that customer.")]}

    return {"messages": [AIMessage(content="Try asking for: top Gambling High Risk customer, loan PDF lookup for a customer id, stock price for $AAPL, or Sentiment: <text>.")]}


def make_final_node(llm: Any) -> Callable[[AgentState], Any]:
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
    async def final_node(state: AgentState):
        if not llm:
            return final_template_node(state)

        user_text = _get_last_user_text(state)
        mode = state.get("mode") or "unknown"

        context = {
            "mode": mode,
            "customer_id": state.get("customer_id"),
            "high_risk_count": state.get("high_risk_count"),
            "doc_query": state.get("doc_query"),
            "income_source": state.get("income_source"),
            "doc_fact_line": state.get("doc_fact_line"),
            "ticker": state.get("ticker"),
            "market_result": state.get("market_result"),
            "sentiment_result": state.get("sentiment_result"),
        }

        excerpt = state.get("loan_excerpt") or ""
        if excerpt:
            context["loan_excerpt"] = excerpt[:1600]

        sys_msg = SystemMessage(
            content=(
                "You are PeakWhale Helm. Answer the user using only the provided context and tool results. "
                "Do not invent customer ids, prices, or document facts. "
                "If something is missing, say what is missing and suggest the next user action."
            )
        )

        user_msg = HumanMessage(
            content=(
                f"User question:\n{user_text}\n\n"
                f"Context JSON:\n{json.dumps(context, ensure_ascii=False, indent=2)}\n\n"
                "Write a clear, concise answer."
            )
        )

        try:
            resp = await llm.ainvoke([sys_msg, user_msg])
            text = getattr(resp, "content", "") or ""
            text = text.strip() or "I could not generate a response."
            return {"messages": [AIMessage(content=text)]}
        except Exception as e:
            logger.warning("Ollama call failed, falling back to template. Details: %s", e)
            return final_template_node(state)

    return final_node


def _should_continue(state: AgentState):
    last = state["messages"][-1]
    tcs = getattr(last, "tool_calls", None) or []
    return "MCPTools" if tcs else "Supervisor"


@dataclass
class GraphRuntime:
    graph: Any
    _client: Any
    _session_ctx: Any
    llm: Any = None
    llm_name: Optional[str] = None

    @staticmethod
    def _find_project_root(start: Path) -> Path:
        candidates = [
            "pyproject.toml",
            "requirements.txt",
            "README.md",
            ".git",
        ]
        cur = start.resolve()
        for _ in range(8):
            for name in candidates:
                if (cur / name).exists():
                    return cur
            if cur.parent == cur:
                break
            cur = cur.parent
        return start.resolve()

    @classmethod
    def _project_root(cls) -> Path:
        return cls._find_project_root(Path(__file__).resolve())

    @classmethod
    def _tools_server_path(cls) -> str:
        env_path = os.environ.get("HELM_MCP_TOOLS_PATH")
        if env_path:
            return env_path

        root = cls._project_root()

        root_tools = root / "tools.py"
        if root_tools.exists():
            return str(root_tools)

        src_tools = root / "src" / "tools.py"
        if src_tools.exists():
            return str(src_tools)

        return str(root / "tools.py")

    @classmethod
    async def create(cls) -> "GraphRuntime":
        from langgraph.graph import StateGraph, START, END
        from langgraph.prebuilt import ToolNode
        from langchain_mcp_adapters.client import MultiServerMCPClient
        from langchain_mcp_adapters.tools import load_mcp_tools

        project_root = cls._project_root()
        tools_path = cls._tools_server_path()

        llm, llm_name = _build_ollama_llm()

        connections = {
            "helm_tools": {
                "transport": os.environ.get("HELM_MCP_TRANSPORT", "stdio"),
                "command": os.environ.get("HELM_MCP_COMMAND", sys.executable),
                "args": [tools_path],
                "cwd": str(project_root),
                "env": {
                    **os.environ,
                    "PYTHONUNBUFFERED": "1",
                    "PYTHONPATH": str(project_root),
                    "HELM_MCP_TRANSPORT": os.environ.get("HELM_MCP_TRANSPORT", "stdio"),
                },
            }
        }

        client = MultiServerMCPClient(connections, tool_name_prefix=False)

        session_ctx = client.session("helm_tools")
        session = await session_ctx.__aenter__()

        mcp_tools = await load_mcp_tools(session, server_name="helm_tools", tool_name_prefix=False)
        mcp_tool_node = ToolNode(mcp_tools)

        workflow = StateGraph(AgentState)
        workflow.add_node("Supervisor", supervisor_node)
        workflow.add_node("Quant", quant_node)
        workflow.add_node("Researcher", researcher_node)
        workflow.add_node("Market", market_node)
        workflow.add_node("Sentiment", sentiment_node)
        workflow.add_node("MCPTools", mcp_tool_node)
        workflow.add_node("Extractor", extractor_node)
        workflow.add_node("Final", make_final_node(llm))

        workflow.add_edge(START, "Supervisor")

        workflow.add_conditional_edges(
            "Supervisor",
            lambda s: s["next"],
            {
                "Quant": "Quant",
                "Researcher": "Researcher",
                "Market": "Market",
                "Sentiment": "Sentiment",
                "Final": "Final",
            },
        )

        workflow.add_conditional_edges("Quant", _should_continue, {"MCPTools": "MCPTools", "Supervisor": "Supervisor"})
        workflow.add_conditional_edges("Researcher", _should_continue, {"MCPTools": "MCPTools", "Supervisor": "Supervisor"})
        workflow.add_conditional_edges("Market", _should_continue, {"MCPTools": "MCPTools", "Supervisor": "Supervisor"})
        workflow.add_conditional_edges("Sentiment", _should_continue, {"MCPTools": "MCPTools", "Supervisor": "Supervisor"})

        workflow.add_edge("MCPTools", "Extractor")
        workflow.add_edge("Extractor", "Supervisor")
        workflow.add_edge("Final", END)

        graph = workflow.compile()
        return cls(graph=graph, _client=client, _session_ctx=session_ctx, llm=llm, llm_name=llm_name)

    async def aclose(self) -> None:
        if self._session_ctx:
            try:
                # Use a timeout for closing to prevent hanging on shutdown
                import anyio
                async with anyio.fail_after(5):
                    await self._session_ctx.__aexit__(None, None, None)
            except Exception as e:
                logger.debug("Error during MCP session cleanup (expected in some async environments): %s", e)
            finally:
                self._session_ctx = None
