"""Rule-based fallback planner. It answers simple questions without an LLM.

It is the safety net when no model is running, and it makes the demo and the
evals work offline. It is not clever:

1. Split the question into parts ("find X, then check Y, and get Z").
2. Match each part to one tool with simple rules: raw SQL, list tables,
   describe a table, a saved query (keywords from helm.toml), a document
   search, a row count, a record lookup by id, a stock price or a sentiment score.
3. Run the tools through the same tool set as the LLM agent, and write a short
   answer with [S#] tags and a Sources list.

A result from an earlier part can feed a later one: in "find the top high risk
customer and check their loan application", the customer id from the SQL result
becomes the file filter for the document search.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Sequence

from .config import SavedQuery
from .documents import terms
from .formatting import markdown_table, parse_markdown_table, shorten
from .sources import Source, error_text, format_sources

ToolCaller = Callable[[str, dict[str, Any]], Awaitable[str]]

FOOTER = (
    "_Answered by the rule-based fallback (no LLM). "
    "It only understands simple questions; connect an LLM for open questions._"
)

# Raw SQL: a query, or a write or admin statement (which the guard then blocks with a
# clear message). A question that only starts with such a word, like "Create a summary
# of the credit policy" or "Update me on ...", is not SQL.
SQL_START = re.compile(
    r"""^\s*\(?\s*(?:
        select\b
      | with\s+(?:recursive\s+)?[\w"]+(?:\s*\([^)]*\))?\s+as\b
      | (?:drop|create(?:\s+or\s+replace)?(?:\s+(?:temp|temporary|unique|persistent))?)\s+
        (?:table|view|schema|index|database|sequence|macro|function|type|secret)\b
      | alter\s+(?:table|view|schema|database|sequence)\b
      | delete\s+from\b
      | insert\s+(?:or\s+\w+\s+)?into\b
      | update\s+[\w".]+\s+set\b
      | merge\s+into\b
      | copy\s+(?:\(|[\w".]+\s+(?:from|to)\b)
      | (?:attach|detach)\s+(?:database\s+)?(?:if\s+(?:not\s+)?exists\s+)?['"\w./:-]+\s*(?:as\b|\(|;|$)
      | (?:install|load|truncate|vacuum|reset)\s+(?:table\s+)?['"\w.-]+\s*;?\s*$
      | (?:checkpoint|vacuum)\s*;?\s*$
      | pragma\s+\w+
      | set\s+(?:session\s+|global\s+|local\s+)?[\w.]+\s*(?:=|to\b)
      | call\s+\w+\s*\(
      | (?:export|import)\s+database\b
      | (?:grant|revoke)\s+[\w\s,]+?\s+on\b
    )""",
    re.IGNORECASE | re.VERBOSE,
)
_VERBS = (
    r"check|get|find|show|analy[sz]e|list|tell|give|fetch|look|search|describe|count|pull|read|"
    r"summari[sz]e|compare|run|what|which|how"
)
CLAUSE_SPLIT = re.compile(
    rf"\s*[;,]\s*(?:and\s+|then\s+)?|\s+then\s+|\s+and\s+(?:then\s+)?(?=(?:{_VERBS})\b)",
    re.IGNORECASE,
)
QUOTED = re.compile(r"(?:(?<=^)|(?<=[\s:(]))[\"'“‘]([^\"'“”‘’]{3,}?)[\"'”’](?=$|[\s.,;:!?)])")
ID_TOKEN = re.compile(r"\b(?=[A-Za-z0-9_-]*\d)(?=[A-Za-z0-9_-]*[A-Za-z])[A-Za-z0-9][A-Za-z0-9_-]{3,63}\b")
TICKER_DOLLAR = re.compile(r"\$([A-Za-z][A-Za-z0-9.\-]{0,9})\b")
STOCK_WORDS = re.compile(r"\b(?:stock|share)\s+price\b|\bprice\s+(?:of|for)\b|\bquote\s+for\b|\bticker\b", re.I)
LIST_TABLES = re.compile(
    r"^\s*(?:what|which|list|show)(?:\s+(?:me|all|the|are|of|available))*\s+tables\b|\bwhat data\b|\bwhich data\b",
    re.I,
)
LIST_DOCS = re.compile(
    r"^\s*(?:what|which|list|show)(?:\s+(?:me|all|the|are|of|available))*\s+(?:documents|files|docs)\b"
    r"(?!.*\b(?:say|says|mention|mentions|about|contain|contains)\b)",
    re.I,
)
DESCRIBE = re.compile(r"\b(?:describe|columns?|schema|fields|structure)\b", re.I)
COUNT = re.compile(r"\bhow many\b|\bnumber of\b|\bcount\b", re.I)
# The noun right after "how many" / "number of" / "count" says which table is counted.
COUNT_OBJECT = re.compile(r"\b(?:how many|number of|count(?: of)?)\s+(?:(?:the|all|of)\s+)*(.*)$")
# Words that may come with an unfiltered count ("How many customers are in the ledger?").
# Any other word is a filter ("... have a risk score above 700"), which the count rule cannot do.
COUNT_FILLER = frozenset(
    """how many number of count the a an all are is there s in on inside our my your this that these
    those total altogether overall rows row records record entries entry items table tables database db
    data dataset ledger do does did we you i have has exist exists stored listed currently now so far
    what whats tell me please can could would show give find get know""".split()
)
SENTIMENT = re.compile(r"\bsentiment\b", re.I)

DOC_WORDS = frozenset(
    """document documents doc docs pdf pdfs file files application applications loan loans policy
    policies contract contracts agreement agreements report reports letter memo memos note notes
    manual handbook stated declare declares declared say says said mention mentions mentioned
    according clause section page pages""".split()
)
# Words that say where to look or what to do, not what to find.
NOT_FOCUS = frozenset(
    terms(" ".join(DOC_WORDS | {"check", "find", "search", "look", "read", "tell", "show", "give", "pull",
                                 "their", "his", "her", "its", "this", "that", "on", "the"}),
          drop_stopwords=False)
)
TICKER_STOP = {"GET", "THE", "FOR", "OF", "PRICE", "STOCK", "SHARE", "QUOTE", "TICKER", "LATEST", "WHAT", "IS"}


def _normalize(text: str) -> str:
    return " ".join(re.sub(r"[^0-9a-z]+", " ", (text or "").lower()).split())


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", (text or "").lower()))


def _singular(word: str) -> str:
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith("ses") or word.endswith("xes"):
        return word[:-2]
    if word.endswith("s") and not word.endswith("ss") and len(word) > 3:
        return word[:-1]
    return word


def _table_mentioned(text: str, tables: Sequence[str]) -> str | None:
    norm = f" {_normalize(text)} "
    for table in tables:
        if any(f" {v} " in norm for v in _table_variants(table)):
            return table
    return None


def _table_variants(table: str) -> set[str]:
    short = table.split(".")[-1].lower()
    variants = {short, _singular(short), short.replace("_", " "), _singular(short).replace("_", " ")}
    return {v for v in variants if len(v) > 2}


def _count_table(text: str, tables: Sequence[str]) -> str | None:
    """The table a count question is about: the noun after "how many", else any table named."""
    match = COUNT_OBJECT.search(_normalize(text))
    if match:
        after = f" {match.group(1)} "
        for table in tables:
            if any(after.startswith(f" {v} ") for v in _table_variants(table)):
                return table
    return _table_mentioned(text, tables)


def _count_has_filter(text: str, table: str) -> bool:
    """True when a count question asks about only some rows (an id, a value, another table ...)."""
    norm = f" {_normalize(text)} "
    for variant in sorted(_table_variants(table), key=len, reverse=True):
        norm = norm.replace(f" {variant} ", " ")
    return any(word not in COUNT_FILLER for word in norm.split())


def _quote_ident(name: str) -> str:
    return ".".join('"' + part.replace('"', '""') + '"' for part in name.split("."))


def _row_text(columns: Sequence[str], row: Sequence[str], only: Sequence[str] | None = None) -> str:
    pairs = [(c, v) for c, v in zip(columns, row) if not only or c in only]
    return ", ".join(f"{c} = {v}" for c, v in pairs)


def _best_line(text: str, focus: set[str]) -> tuple[str, int]:
    """Pick the line that shares the most words with the question.

    On a tie, a "Label: value" line wins, then the earlier line.
    """
    best, best_key = "", (-1, -1)
    for line in (text or "").splitlines():
        stripped = line.strip().lstrip("#").strip()
        if not stripped:
            continue
        key = (len(focus & set(terms(stripped))), 1 if ":" in stripped else 0)
        if key > best_key:
            best, best_key = stripped, key
    return best, max(best_key[0], 0)


def _ids(text: str) -> list[str]:
    """Id-like tokens (letters and digits, like 2631d00b), without our QUOTE placeholders."""
    return [m.group(0) for m in ID_TOKEN.finditer(text or "") if not re.fullmatch(r"QUOTE\d+", m.group(0))]


@dataclass
class Part:
    text: str
    kind: str | None = None
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass
class FallbackResult:
    answer: str
    sources: list[Source]


class FallbackPlanner:
    def __init__(
        self,
        call_tool: ToolCaller,
        tool_names: Sequence[str],
        saved_queries: Sequence[SavedQuery] = (),
    ):
        self._call_tool = call_tool
        self.tool_names = set(tool_names)
        self.saved_queries = list(saved_queries)

    # ------------------------------------------------------------------ helpers

    async def _call(self, run: "_Run", tool: str, args: dict[str, Any]) -> Source:
        source = Source(f"S{len(run.sources) + 1}", tool, dict(args))
        try:
            source.output = await self._call_tool(tool, dict(args))
        except Exception as exc:  # tool errors become part of the answer
            source.output = error_text(exc)
            source.error = True
        run.sources.append(source)
        return source

    async def _document_names(self, run: "_Run") -> list[str]:
        """File names in the documents folder, lower case (looked up once per question).

        Only used to decide on a file filter, so it is not listed as a source.
        """
        if run.documents is None:
            try:
                output = await self._call_tool("list_documents", {})
            except Exception:
                output = ""
            columns, rows = parse_markdown_table(output)
            run.documents = [r[0].lower() for r in rows] if columns and columns[0] == "file" else []
        return run.documents

    async def _tables(self, run: "_Run", force: bool = False) -> list[str]:
        """Table names from list_tables (called once per question)."""
        if run.tables is None or force:
            source = await self._call(run, "list_tables", {})
            columns, rows = parse_markdown_table(source.output) if not source.error else ([], [])
            run.tables = [r[0] for r in rows] if columns and columns[0] == "table" else []
        return run.tables

    def _saved_match(self, text: str) -> SavedQuery | None:
        norm = f" {_normalize(text)} "
        for saved in self.saved_queries:
            if any(f" {_normalize(k)} " in norm for k in saved.keywords if _normalize(k)):
                return saved
        return None

    # ------------------------------------------------------------------ planning

    def _split(self, question: str) -> tuple[list[str], list[str]]:
        quotes: list[str] = []

        def keep(match: re.Match) -> str:
            quotes.append(match.group(1))
            return f"QUOTE{len(quotes) - 1}"

        text = QUOTED.sub(keep, question.strip())
        parts = [p.strip(" .?!") for p in CLAUSE_SPLIT.split(text) if p and p.strip(" .?!")]
        return parts, quotes

    def _classify(self, text: str, quotes: list[str]) -> Part:
        words = _words(text)
        part = Part(text)
        if SENTIMENT.search(text):
            refs = [quotes[int(n)] for n in re.findall(r"QUOTE(\d+)", text)]
            if refs:
                payload = refs[0]
            else:
                after = re.split(r"\bsentiment\b(?:\s+(?:of|for|on|in))?\s*:?\s*", text, maxsplit=1, flags=re.I)
                payload = after[1] if len(after) > 1 else ""
            part.kind, part.detail = "sentiment", {"text": payload.strip()}
            return part
        ticker = TICKER_DOLLAR.search(text)
        if ticker or STOCK_WORDS.search(text):
            symbol = ticker.group(1) if ticker else None
            if not symbol:
                for match in re.finditer(r"\b(?:of|for)\s+([A-Za-z][A-Za-z0-9.\-]{0,9})\b", text):
                    if match.group(1).upper() not in TICKER_STOP:
                        symbol = match.group(1)
                        break
            part.kind, part.detail = "stock", {"ticker": (symbol or "").upper()}
            return part
        if LIST_TABLES.search(text):
            part.kind = "tables"
            return part
        if LIST_DOCS.search(text):
            part.kind = "documents"
            return part
        if words & DOC_WORDS:
            part.kind = "docs"
            return part
        saved = self._saved_match(text)
        if saved:
            part.kind, part.detail = "saved", {"saved": saved}
            return part
        if DESCRIBE.search(text):
            part.kind = "describe"
        elif COUNT.search(text):
            part.kind = "count"
        elif _ids(text):
            part.kind = "lookup"
        return part

    # ------------------------------------------------------------------ running

    async def run(self, question: str) -> FallbackResult:
        run = _Run(question=question)
        question = (question or "").strip()
        if not question:
            return FallbackResult(self._help(), [])

        if SQL_START.match(question) and not question.endswith("?"):
            await self._do_sql(run, question)
            return self._finish(run)

        texts, quotes = self._split(question)
        parts = [self._classify(t, quotes) for t in texts]
        generic = not any(p.kind for p in parts)
        if generic:
            if not terms(question):
                return FallbackResult(self._help(), [])
            parts = [Part(question, "docs")]
        # Parts with no rule of their own add context to the document search.
        loose = " ".join(p.text for p in parts if p.kind is None)

        for part in parts:
            if part.kind == "sentiment":
                await self._do_sentiment(run, part)
            elif part.kind == "stock":
                await self._do_stock(run, part)
            elif part.kind == "tables":
                await self._tables(run, force=True)
                self._say_tables(run, run.sources[-1])
            elif part.kind == "documents":
                source = await self._call(run, "list_documents", {})
                self._say_plain(run, source, "Documents")
            elif part.kind == "docs":
                await self._do_docs(run, part, loose)
            elif part.kind == "saved":
                await self._do_saved(run, part)
            elif part.kind in ("describe", "count", "lookup"):
                await self._do_table_part(run, part)
        nothing_found = generic and all(s.tool == "search_documents" and not s.hits() for s in run.sources)
        if not run.lines or nothing_found:
            return FallbackResult(self._help(), run.sources)
        return self._finish(run)

    def _finish(self, run: "_Run") -> FallbackResult:
        body = "\n\n".join(run.lines)
        sources = format_sources(run.sources)
        answer = body + (f"\n\n{sources}" if sources else "") + f"\n\n{FOOTER}"
        return FallbackResult(answer, run.sources)

    def _help(self) -> str:
        return (
            "I could not match this question to a tool without an LLM. Simple questions work best, for example:\n"
            "- What tables are in the database?\n"
            "- Describe the <table> table\n"
            "- How many <table> are there?\n"
            "- SELECT ... (one read-only SQL query)\n"
            "- What does the <document> say about <topic>?\n\n"
            "For open questions, start Ollama or set an OpenAI-compatible endpoint (see the README).\n\n"
            + FOOTER
        )

    # ------------------------------------------------------------------ steps

    def _fail(self, run: "_Run", source: Source, what: str) -> None:
        run.lines.append(f"Could not {what}: {source.output} [{source.id}]")

    def _say_plain(self, run: "_Run", source: Source, title: str) -> None:
        if source.error:
            self._fail(run, source, f"get the {title.lower()}")
        else:
            run.lines.append(f"{title} [{source.id}]:\n\n{source.output}")

    def _say_tables(self, run: "_Run", source: Source) -> None:
        if source.error:
            self._fail(run, source, "list the tables")
            return
        columns, rows = parse_markdown_table(source.output)
        if not rows:
            run.lines.append(f"The database has no tables [{source.id}].")
            return
        items = ", ".join(f"{r[0]} ({r[2]} rows)" if len(r) > 2 else r[0] for r in rows)
        run.lines.append(f"The database has {len(rows)} tables: {items} [{source.id}].")

    async def _do_sql(self, run: "_Run", sql: str) -> None:
        source = await self._call(run, "query", {"sql": sql})
        if source.error:
            self._fail(run, source, "run the query")
            return
        run.lines.append(f"Query result [{source.id}]:\n\n{source.output}")

    async def _do_sentiment(self, run: "_Run", part: Part) -> None:
        if "analyze_sentiment" not in self.tool_names:
            run.lines.append(
                "The sentiment tool is turned off. Set HELM_ENABLE_SENTIMENT=1 (needs `uv sync --extra ml`)."
            )
            return
        text = part.detail.get("text") or ""
        if not text:
            run.lines.append("Tell me the text to score in quotes, like: sentiment of 'Profits beat forecasts'.")
            return
        source = await self._call(run, "analyze_sentiment", {"text": text})
        if source.error:
            self._fail(run, source, "score the sentiment")
        else:
            run.lines.append(f'{source.output} for "{shorten(text, 80)}" [{source.id}]')

    async def _do_stock(self, run: "_Run", part: Part) -> None:
        if "get_stock_price" not in self.tool_names:
            run.lines.append("The stock price tool is turned off. Set HELM_ENABLE_MARKET=1 to turn it on.")
            return
        ticker = part.detail.get("ticker")
        if not ticker:
            run.lines.append("Tell me the ticker, like $AAPL, and I will get the price.")
            return
        source = await self._call(run, "get_stock_price", {"ticker": ticker})
        if source.error:
            self._fail(run, source, f"get the price for {ticker}")
        else:
            run.lines.append(" | ".join(source.output.splitlines()) + f" [{source.id}]")

    async def _do_saved(self, run: "_Run", part: Part) -> None:
        saved: SavedQuery = part.detail["saved"]
        source = await self._call(run, "query", {"sql": saved.sql})
        if source.error:
            self._fail(run, source, f"run the saved query {saved.name}")
            return
        columns, rows = parse_markdown_table(source.output)
        if not rows:
            run.lines.append(f"The saved query {saved.name} returned no rows [{source.id}].")
            return
        wanted = _ids(part.text)
        if wanted:
            key = wanted[0].lower()
            match = [r for r in rows if any(cell.lower() == key for cell in r)]
            if match:
                run.lines.append(f"For {wanted[0]}: {_row_text(columns, match[0])} [{source.id}].")
                run.entity = wanted[0]
            else:
                run.lines.append(
                    f"{wanted[0]} does not appear in the result of the saved query {saved.name} [{source.id}]."
                )
            return
        top = rows[0]
        run.entity = next((cell for cell in top if ID_TOKEN.fullmatch(cell)), top[0])
        text = f"Top result of the saved query {saved.name}: {_row_text(columns, top)} [{source.id}]."
        if len(rows) > 1:
            text += "\n\n" + markdown_table(columns, rows[:5])
            if len(rows) > 5:
                text += f"\n\n({len(rows)} rows in total; showing 5.)"
        run.lines.append(text)

    async def _do_table_part(self, run: "_Run", part: Part) -> None:
        tables = await self._tables(run)
        table = _count_table(part.text, tables) if part.kind == "count" else _table_mentioned(part.text, tables)
        if table is None and part.kind == "lookup":
            # An id but no table: look for it in the documents instead.
            await self._do_docs(run, part, "")
            return
        if table is None:
            known = ", ".join(tables) or "none"
            run.lines.append(f'I could not tell which table "{shorten(part.text, 60)}" is about. Tables: {known}.')
            return
        if part.kind == "describe":
            source = await self._call(run, "describe_table", {"table": table})
            if source.error:
                self._fail(run, source, f"describe {table}")
                return
            columns, rows = parse_markdown_table(source.output)
            listed = ", ".join(f"{r[0]} ({r[1]})" for r in rows)
            run.lines.append(f"Table {table} has {len(rows)} columns: {listed} [{source.id}].")
        elif part.kind == "count":
            if _count_has_filter(part.text, table):
                # "How many transactions does customer X have?" is not the row count of a table.
                run.lines.append(
                    "Without an LLM I can only count all rows of a table, and this question asks about some "
                    "of the rows only, so I did not guess. Connect an LLM, or send one SQL query, for example: "
                    f"SELECT COUNT(*) FROM {_quote_ident(table)} WHERE ..."
                )
                return
            source = await self._call(run, "query", {"sql": f"SELECT COUNT(*) AS row_count FROM {_quote_ident(table)}"})
            if source.error:
                self._fail(run, source, f"count the rows in {table}")
                return
            _, rows = parse_markdown_table(source.output)
            run.lines.append(f"The table {table} has {rows[0][0]} rows [{source.id}].")
        else:
            await self._do_lookup(run, part, table)

    async def _do_lookup(self, run: "_Run", part: Part, table: str) -> None:
        value = _ids(part.text)[0]
        described = await self._call(run, "describe_table", {"table": table})
        if described.error:
            self._fail(run, described, f"read the columns of {table}")
            return
        _, column_rows = parse_markdown_table(described.output)
        names = [r[0] for r in column_rows]
        singular = _singular(table.split(".")[-1].lower())
        id_columns = [n for n in names if n.lower() == f"{singular}_id"] + [n for n in names if n.lower() == "id"]
        id_columns += [n for n in names if n.lower().endswith("_id") and n not in id_columns]
        if not id_columns:
            run.lines.append(f"Table {table} has no id column to look up {value} [{described.id}].")
            return
        column = id_columns[0]
        safe = value.replace("'", "''")
        sql = f"SELECT * FROM {_quote_ident(table)} WHERE CAST({_quote_ident(column)} AS VARCHAR) = '{safe}' LIMIT 5"
        source = await self._call(run, "query", {"sql": sql})
        if source.error:
            self._fail(run, source, f"look up {value}")
            return
        columns, rows = parse_markdown_table(source.output)
        if not rows:
            run.lines.append(f"No row in {table} has {column} = {value} [{source.id}].")
            return
        asked = [c for c in columns if c != column and _words(c.replace("_", " ")) & _words(part.text)]
        run.entity = value
        found = _row_text(columns, rows[0], asked or None)
        run.lines.append(f"{table} row where {column} = {value}: {found} [{source.id}].")

    async def _do_docs(self, run: "_Run", part: Part, loose: str) -> None:
        explicit = _ids(part.text)
        entity = run.entity
        mention = None  # an id that the passage itself must contain
        if explicit:
            # An id-like word (2631d00b, but also COVID-19 or 1099-K) filters the files only when
            # it is part of a file name. Otherwise it stays a search word, and the passage must have it.
            names = await self._document_names(run)
            in_names = [t for t in explicit if any(t.lower() in name for name in names)]
            entity = in_names[0] if in_names else entity
            mention = None if in_names else explicit[0]
        query = re.sub(r"QUOTE\d+", " ", part.text)
        if entity:
            query = re.sub(re.escape(entity), " ", query, flags=re.I)
        query = " ".join(query.split()) or part.text
        args: dict[str, Any] = {"query": query, "top_k": 5 if mention else 3}
        if entity:
            args["source_filter"] = entity
        source = await self._call(run, "search_documents", args)
        if source.error:
            self._fail(run, source, "search the documents")
            return
        hits = list(zip(source.hits(), _hit_texts(source.output)))
        if mention:
            hits = [(label, text) for label, text in hits if mention.lower() in text.lower()]
            if not hits:
                run.lines.append(f"No document passage mentions {mention} [{source.id}].")
                return
        if not hits:
            where = f" in files for {entity}" if entity else ""
            run.lines.append(f'No document passage matched "{query}"{where} [{source.id}].')
            return
        label, text = hits[0]
        focus = set(terms(f"{part.text} {loose}")) - NOT_FOCUS - set(terms(entity or ""))
        line, score = _best_line(text, focus)
        if score == 0:
            line = shorten(text, 300)
        run.lines.append(f'From {label}: "{line}" [{source.id}]')


def _hit_texts(output: str) -> list[str]:
    """The text of each hit ([1], [2], ...) in a search_documents result."""
    pattern = r"^\[\d+\] .+?, score -?[\d.]+\n(.*?)(?=^\[\d+\] |\Z)"
    return [m.group(1).strip() for m in re.finditer(pattern, output, re.S | re.M)]


@dataclass
class _Run:
    question: str
    sources: list[Source] = field(default_factory=list)
    lines: list[str] = field(default_factory=list)
    tables: list[str] | None = None
    documents: list[str] | None = None
    entity: str | None = None
