import operator
import re
import uuid
from typing import Annotated, TypedDict, List, Optional

from langchain_core.messages import BaseMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

try:
    from src.tools import query_ledger, get_stock_price, analyze_sentiment, search_loan_documents
except ImportError:
    from tools import query_ledger, get_stock_price, analyze_sentiment, search_loan_documents


quant_tools = [query_ledger]
researcher_tools = [get_stock_price, analyze_sentiment, search_loan_documents]


class AgentState(TypedDict, total=False):
    messages: Annotated[List[BaseMessage], operator.add]
    next: str

    last_user_text: Optional[str]
    mode: Optional[str]

    customer_id: Optional[str]
    high_risk_count: Optional[int]

    doc_query: Optional[str]
    loan_excerpt: Optional[str]
    income_source: Optional[str]
    doc_fact_line: Optional[str]

    quant_attempts: int
    research_attempts: int


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


def _detect_mode(user_text: str) -> str:
    t = (user_text or "").lower()

    has_quant_label = "quant:" in t
    has_researcher_label = "researcher:" in t

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
            "do not",
        ]
    )

    if has_quant_label and has_researcher_label:
        return "top_then_loan"
    if wants_top and wants_loan:
        return "top_then_loan"
    if wants_top:
        return "top_risk"
    if wants_loan:
        return "loan_query"
    return "unknown"


def _extract_doc_query(user_text: str) -> Optional[str]:
    t = user_text or ""
    tl = t.lower()

    if "researcher:" in tl:
        seg = t[tl.index("researcher:") + len("researcher:") :].strip()
    else:
        seg = t.strip()

    seg_l = seg.lower()

    # Priority 1: explicit income source intent anywhere in the request
    if "income source" in seg_l or "source of income" in seg_l or "stated income source" in seg_l:
        return "income source"

    # Priority 2: if user says "then ... for X", capture X
    m = re.search(r"(?i)\bthen\b.*?\bfor\b\s+(.+)$", seg)
    if m:
        candidate = m.group(1).strip().strip(".")
        if candidate:
            candidate_l = candidate.lower()
            if "income source" in candidate_l or "source of income" in candidate_l:
                return "income source"
            return candidate

    # Priority 3: quoted strings, but ignore quotes that look like transaction categories
    quoted = re.findall(r"['\"]([^'\"]+)['\"]", seg)
    for q in quoted:
        ql = q.lower()
        if "gambling" in ql or "high risk" in ql or "risk" in ql:
            continue
        return q.strip()

    # Fallback: strip common verbs and return remaining phrase
    cleaned = re.sub(r"(?i)^\s*(confirm|check|pull|search|find|retrieve|tell me)\b\s*", "", seg).strip()

    # Polish: remove filler phrases that hurt retrieval quality
    cleaned = re.sub(r"(?i)\bthe line for\b", "", cleaned).strip()
    cleaned = re.sub(r"(?i)\bline for\b", "", cleaned).strip()
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

    header_idx = None
    for i, ln in enumerate(lines):
        if "customer_id" in ln and ("high_risk_count" in ln or "count" in ln.lower()):
            header_idx = i
            break

    if header_idx is not None and header_idx + 1 < len(lines):
        row = lines[header_idx + 1]
        parts = row.split()
        if len(parts) >= 2 and re.fullmatch(r"[a-f0-9]{8}", parts[0].lower()):
            cid = parts[0].lower()
            cnt = int(parts[1]) if parts[1].isdigit() else None
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

        if "do not engage" in q or "speculative trading" in q:
            if "do not engage" in lnl or "speculative trading" in lnl:
                return ln.strip()

        if "annual income" in q:
            if lnl.startswith("annual income"):
                return ln.strip()

        if "income source" in q:
            if "stated income source" in lnl or lnl.startswith("income source"):
                return ln.strip()

    return None


def supervisor_node(state: AgentState):
    user_text = _get_last_user_text(state)
    mode = _detect_mode(user_text)

    is_new_request = user_text != (state.get("last_user_text") or "")
    updates: AgentState = {"mode": mode, "last_user_text": user_text}

    # Reset per new user request so prompts behave independently
    if is_new_request:
        updates["quant_attempts"] = 0
        updates["research_attempts"] = 0
        updates["doc_query"] = None
        updates["loan_excerpt"] = None
        updates["income_source"] = None
        updates["doc_fact_line"] = None

        if mode in ["top_risk", "top_then_loan"]:
            updates["customer_id"] = None
            updates["high_risk_count"] = None

    explicit_cid = _extract_customer_id_from_text(user_text)
    if explicit_cid:
        updates["customer_id"] = explicit_cid

    doc_query = _extract_doc_query(user_text)
    if doc_query:
        updates["doc_query"] = doc_query

    customer_id = updates.get("customer_id", state.get("customer_id"))
    has_customer = bool(customer_id)

    has_count = (
        updates.get("high_risk_count") if "high_risk_count" in updates else state.get("high_risk_count")
    ) is not None

    has_doc = bool(updates.get("loan_excerpt", None) or state.get("loan_excerpt", None))
    has_income = bool(updates.get("income_source", None) or state.get("income_source", None))

    quant_attempts = updates.get("quant_attempts", state.get("quant_attempts", 0))
    research_attempts = updates.get("research_attempts", state.get("research_attempts", 0))

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

        if research_attempts >= 3:
            return {"next": "Final", **updates}
        return {"next": "Researcher", **updates}

    if mode == "top_then_loan":
        if not (has_customer and has_count):
            if quant_attempts >= 2:
                return {"next": "Final", **updates}
            return {"next": "Quant", **updates}

        if has_income or has_doc:
            return {"next": "Final", **updates}

        if research_attempts >= 3:
            return {"next": "Final", **updates}
        return {"next": "Researcher", **updates}

    return {"next": "Final", **updates}


def quant_node(state: AgentState):
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
    return {"messages": [msg], "research_attempts": state.get("research_attempts", 0) + 1}


def extractor_node(state: AgentState):
    if not state.get("messages"):
        return {}

    last_msg = state["messages"][-1]
    if getattr(last_msg, "type", "") != "tool":
        return {}

    content = getattr(last_msg, "content", "") or ""
    updates: AgentState = {}

    if "customer_id" in content and ("high_risk_count" in content or "count" in content.lower()):
        cid, cnt = _extract_customer_id_and_count(content)
        if cid:
            updates["customer_id"] = cid
        if cnt is not None:
            updates["high_risk_count"] = cnt

    if content.startswith("Document Excerpt for"):
        updates["loan_excerpt"] = content

        inc = _extract_income_source(content)
        if inc:
            updates["income_source"] = inc

        fact = _extract_fact_line(content, state.get("doc_query"))
        if fact:
            updates["doc_fact_line"] = fact

    return updates


def final_node(state: AgentState):
    mode = state.get("mode") or "unknown"
    customer_id = state.get("customer_id")
    cnt = state.get("high_risk_count")

    doc_query = state.get("doc_query") or ""
    income_source = state.get("income_source")
    excerpt = state.get("loan_excerpt")
    fact_line = state.get("doc_fact_line")

    if mode == "top_risk":
        if not customer_id:
            return {"messages": [AIMessage(content="I could not find a customer for the Gambling or High Risk category.")]}
        line = f"Customer with the most 'Gambling/High Risk' transactions: **{customer_id}**"
        if cnt is not None:
            line += f" (**{cnt}** transactions)."
        else:
            line += "."
        return {"messages": [AIMessage(content=line)]}

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

    if customer_id and cnt is not None and income_source:
        text = (
            f"Customer with the most 'Gambling/High Risk' transactions: **{customer_id}** (**{cnt}** transactions).\n"
            f"Income source stated on their loan application: **{income_source}**"
        )
        return {"messages": [AIMessage(content=text)]}

    return {"messages": [AIMessage(content="Ask me to find the top Gambling or High Risk customer, or to check a loan PDF for a specific fact.")]}    


tool_node = ToolNode(quant_tools + researcher_tools)

workflow = StateGraph(AgentState)
workflow.add_node("Supervisor", supervisor_node)
workflow.add_node("Quant", quant_node)
workflow.add_node("Researcher", researcher_node)
workflow.add_node("Tools", tool_node)
workflow.add_node("Extractor", extractor_node)
workflow.add_node("Final", final_node)

workflow.add_edge(START, "Supervisor")

workflow.add_conditional_edges(
    "Supervisor",
    lambda s: s["next"],
    {"Quant": "Quant", "Researcher": "Researcher", "Final": "Final"},
)


def _should_continue(state: AgentState):
    last = state["messages"][-1]
    tcs = getattr(last, "tool_calls", None) or []
    return "Tools" if tcs else "Supervisor"


workflow.add_conditional_edges("Quant", _should_continue, {"Tools": "Tools", "Supervisor": "Supervisor"})
workflow.add_conditional_edges("Researcher", _should_continue, {"Tools": "Tools", "Supervisor": "Supervisor"})

workflow.add_edge("Tools", "Extractor")
workflow.add_edge("Extractor", "Supervisor")
workflow.add_edge("Final", END)

app = workflow.compile()
