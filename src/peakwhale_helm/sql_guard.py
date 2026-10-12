"""Read-only SQL guard.

Every SQL string that comes from a user or an LLM goes through :func:`check_sql`
before it reaches a database. The rules are simple on purpose:

* one statement only (a single trailing ``;`` is fine)
* it must start with ``SELECT`` or ``WITH``
* no write or admin keywords anywhere (``DROP``, ``ATTACH``, ``COPY``, ``INSTALL`` ...)
* no functions that read files, call the network or run other SQL
  (``read_csv``, ``glob``, ``query`` ...), and no ``FROM 'some/file.csv'``
* comments are removed before the checks, so they cannot hide a statement

The guard is one layer. The database connection itself is also opened read-only,
and :func:`limit_rows` caps the number of rows that come back.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

__all__ = ["SQLGuardError", "check_sql", "limit_rows", "BLOCKED_KEYWORDS", "BLOCKED_FUNCTIONS"]


class SQLGuardError(ValueError):
    """The SQL was blocked. The message says why, in plain words."""


ALLOWED_START = {"SELECT", "WITH"}

# Statement words that have no place inside a read-only query.
BLOCKED_KEYWORDS = frozenset(
    {
        "INSERT", "UPDATE", "DELETE", "MERGE", "UPSERT", "TRUNCATE",
        "CREATE", "ALTER", "DROP", "RENAME",
        "ATTACH", "DETACH", "COPY", "EXPORT", "IMPORT", "INSTALL", "LOAD",
        "PRAGMA", "SET", "RESET", "CALL", "EXECUTE", "PREPARE", "DEALLOCATE",
        "VACUUM", "CHECKPOINT", "GRANT", "REVOKE",
    }
)

# Functions that read files, reach the network, change settings or run other SQL.
BLOCKED_FUNCTIONS = frozenset(
    {
        # DuckDB file and network readers
        "read_csv", "read_csv_auto", "read_parquet", "parquet_scan", "parquet_metadata",
        "parquet_schema", "parquet_file_metadata", "parquet_kv_metadata", "parquet_bloom_probe",
        "read_json", "read_json_auto", "read_json_objects", "read_json_objects_auto",
        "read_ndjson", "read_ndjson_auto", "read_ndjson_objects", "read_text", "read_blob",
        "glob", "sniff_csv", "iceberg_scan", "iceberg_metadata", "iceberg_snapshots",
        "delta_scan", "read_xlsx", "st_read", "st_read_meta",
        # DuckDB scanners for other databases, and SQL-in-a-string
        "sqlite_scan", "sqlite_attach", "postgres_scan", "postgres_scan_pushdown",
        "postgres_query", "postgres_execute", "postgres_attach", "mysql_query", "mysql_execute",
        "query", "query_table", "duckdb_secrets", "which_secret", "load_extension",
        # SQLite shell helpers
        "readfile", "writefile", "edit", "fts3_tokenizer",
        # PostgreSQL server-side file and admin functions
        "pg_read_file", "pg_read_binary_file", "pg_ls_dir", "pg_stat_file", "lo_import",
        "lo_export", "lo_get", "lo_put", "lo_from_bytea", "dblink", "dblink_exec",
        "dblink_connect", "pg_sleep", "set_config", "pg_terminate_backend",
        "pg_cancel_backend", "pg_reload_conf", "pg_logdir_ls",
    }
)


@dataclass(frozen=True)
class _Token:
    kind: str  # word | quoted | string | semicolon | lparen | rparen | comma | other
    text: str
    start: int


_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_$]*")
_CLAUSE_WORDS = frozenset(
    {"SELECT", "FROM", "JOIN", "WHERE", "GROUP", "HAVING", "ORDER", "LIMIT", "ON", "USING",
     "WINDOW", "QUALIFY", "UNION", "EXCEPT", "INTERSECT", "VALUES", "OFFSET"}
)


def _tokenize(sql: str) -> tuple[list[_Token], str]:
    """Split SQL into tokens and return them with a copy of the SQL without comments.

    The tokenizer is strict on purpose. Anything it is not sure about is treated
    as code, never as a string or a comment, so it can only block too much, not
    too little.
    """
    tokens: list[_Token] = []
    out: list[str] = []
    i, n = 0, len(sql)
    while i < n:
        ch = sql[i]
        nxt = sql[i + 1] if i + 1 < n else ""
        if ch == "-" and nxt == "-":
            end = sql.find("\n", i)
            end = n if end == -1 else end
            out.append(" ")
            i = end
            continue
        if ch == "/" and nxt == "*":
            end = sql.find("*/", i + 2)
            if end == -1:
                raise SQLGuardError("Blocked: a /* comment is not closed.")
            out.append(" ")
            i = end + 2
            continue
        if ch in ("'", '"'):
            j = i + 1
            while True:
                j = sql.find(ch, j)
                if j == -1:
                    what = "text value (')" if ch == "'" else 'quoted name (")'
                    raise SQLGuardError(f"Blocked: a {what} is not closed.")
                if j + 1 < n and sql[j + 1] == ch:  # doubled quote = escaped quote
                    j += 2
                    continue
                break
            text = sql[i : j + 1]
            tokens.append(_Token("string" if ch == "'" else "quoted", text, i))
            out.append(text)
            i = j + 1
            continue
        if ch == ";":
            tokens.append(_Token("semicolon", ch, i))
            out.append(ch)
            i += 1
            continue
        if ch in "(),":
            kind = {"(": "lparen", ")": "rparen", ",": "comma"}[ch]
            tokens.append(_Token(kind, ch, i))
            out.append(ch)
            i += 1
            continue
        match = _WORD.match(sql, i)
        if match:
            tokens.append(_Token("word", match.group(0), i))
            out.append(match.group(0))
            i = match.end()
            continue
        if not ch.isspace():
            tokens.append(_Token("other", ch, i))
        out.append(ch)
        i += 1
    return tokens, "".join(out)


def _normalize_quotes(sql: str) -> str:
    # Chat apps often turn quotes into "smart" quotes. Make them plain again.
    return (
        sql.replace("\u2018", "'")
        .replace("\u2019", "'")
        .replace("\u201c", "'")
        .replace("\u201d", "'")
    )


def check_sql(sql: str) -> str:
    """Return a cleaned, single read-only statement, or raise :class:`SQLGuardError`.

    The returned SQL has comments removed and no trailing semicolon.
    """
    if sql is None or not str(sql).strip():
        raise SQLGuardError("Blocked: the query is empty. Send one SELECT statement.")
    sql = _normalize_quotes(str(sql))
    if "\x00" in sql:
        raise SQLGuardError("Blocked: the query contains a NUL character.")

    tokens, cleaned = _tokenize(sql)

    # Drop trailing semicolons, then make sure no other semicolon is left.
    while tokens and tokens[-1].kind == "semicolon":
        tokens.pop()
        cleaned = cleaned[: cleaned.rfind(";")]
    if not tokens:
        raise SQLGuardError("Blocked: the query is empty. Send one SELECT statement.")
    if any(t.kind == "semicolon" for t in tokens):
        raise SQLGuardError(
            "Blocked: only one statement is allowed. Remove the extra ';' and send a single SELECT."
        )

    first = next((t for t in tokens if t.kind != "lparen"), None)
    if first is None or first.kind != "word" or first.text.upper() not in ALLOWED_START:
        found = first.text if first is not None else "nothing"
        raise SQLGuardError(
            f"Blocked: only read-only queries that start with SELECT or WITH are allowed "
            f"(found: {found}). This tool cannot change data or settings."
        )

    # The clause we are in (FROM, WHERE, ...) at each bracket depth. Used to spot
    # DuckDB "replacement scans" such as  FROM t, 'secret.csv'.
    clause_stack: list[str] = [""]
    previous: _Token | None = None
    for index, token in enumerate(tokens):
        upper = token.text.upper()
        if token.kind == "lparen":
            clause_stack.append("")
        elif token.kind == "rparen" and len(clause_stack) > 1:
            clause_stack.pop()
        elif token.kind == "word" and upper in _CLAUSE_WORDS:
            clause_stack[-1] = upper
        if token.kind == "word" and upper in BLOCKED_KEYWORDS:
            raise SQLGuardError(
                f"Blocked: the keyword {upper} is not allowed in a read-only query. "
                f'If it is a column name, put it in double quotes, like "{token.text.lower()}".'
            )
        nxt = tokens[index + 1] if index + 1 < len(tokens) else None
        if token.kind in ("word", "quoted") and nxt is not None and nxt.kind == "lparen":
            name = token.text.strip('"').lower()
            if name in BLOCKED_FUNCTIONS:
                raise SQLGuardError(
                    f"Blocked: the function {name}() can read files, reach other systems or run "
                    "other SQL. Query the tables listed by list_tables instead."
                )
        if token.kind == "string" and previous is not None:
            used_as_table = (previous.kind == "word" and previous.text.upper() in ("FROM", "JOIN")) or (
                previous.kind == "comma" and clause_stack[-1] in ("FROM", "JOIN")
            )
            if used_as_table:
                raise SQLGuardError(
                    "Blocked: reading a file or URL with FROM '...' is not allowed. "
                    "Query the tables listed by list_tables instead."
                )
        previous = token

    return cleaned.strip()


def limit_rows(sql: str, max_rows: int) -> str:
    """Wrap a checked query so the database returns at most ``max_rows + 1`` rows.

    One extra row is fetched so the caller can tell the user that the result was cut.
    """
    if max_rows < 1:
        raise ValueError("max_rows must be at least 1")
    return f"SELECT * FROM (\n{sql}\n) AS helm_result LIMIT {int(max_rows) + 1}"
