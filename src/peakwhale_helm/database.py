"""Read-only access to the configured database.

Supported sources:

* ``duckdb``   - a DuckDB file, opened read-only
* ``sqlite``   - a SQLite file, opened read-only (Python's built-in sqlite3)
* ``files``    - a folder (or one file) of CSV, TSV or Parquet files, loaded into
                 an in-memory DuckDB as one view per file
* ``postgres`` - experimental: a PostgreSQL database attached read-only through
                 DuckDB's postgres extension

Every user query goes through :mod:`peakwhale_helm.sql_guard` first. On top of that:

* DuckDB's own parser checks that the final SQL is exactly one SELECT statement
* DuckDB and SQLite files are opened read-only, PostgreSQL is attached read-only
* DuckDB can read only the data files themselves (and a private, empty temp folder)
* settings are locked, and a timer stops queries that run too long
"""

from __future__ import annotations

import contextlib
import logging
import re
import sqlite3
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence
from urllib.request import pathname2url

from .config import FILE_TABLE_SUFFIXES, Settings
from .formatting import markdown_table
from .sql_guard import SQLGuardError, check_sql, limit_rows

logger = logging.getLogger("helm.database")

_HIDDEN_SCHEMAS = ("information_schema", "pg_catalog")
_DEFAULT_SCHEMAS = ("main", "public")


class DatabaseError(Exception):
    """A problem the user can fix. The message is shown as is."""


@dataclass(frozen=True)
class TableInfo:
    name: str  # name to use in SQL (schema.table when not in the default schema)
    schema: str
    table: str
    kind: str  # table | view
    rows: int | None = None


@dataclass(frozen=True)
class QueryResult:
    columns: list[str]
    rows: list[tuple[Any, ...]]
    truncated: bool
    max_rows: int

    def to_markdown(self) -> str:
        if not self.columns:
            return "(no columns)\n\nRows: 0"
        if not self.rows:
            return markdown_table(self.columns, []) + "\n\n(no rows)\n\nRows: 0"
        text = markdown_table(self.columns, self.rows) + f"\n\nRows: {len(self.rows)}"
        if self.truncated:
            text += (
                f" (stopped at the row limit of {self.max_rows}; "
                "add filters, GROUP BY or LIMIT to get a smaller result)"
            )
        return text


def quote_ident(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


def _sql_text(value: str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _sqlite_authorizer(action: int, arg1: Any, arg2: Any, db_name: Any, source: Any) -> int:
    """SQLite engine-level guard for user queries: allow reads only."""
    allowed = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, getattr(sqlite3, "SQLITE_RECURSIVE", 33)}
    if action in allowed:
        return sqlite3.SQLITE_OK
    if action == sqlite3.SQLITE_FUNCTION:
        name = str(arg2 or "").lower()
        if name in {"load_extension", "readfile", "writefile", "edit", "fts3_tokenizer"}:
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK
    return sqlite3.SQLITE_DENY


class Database:
    """Read-only database facade used by the tools."""

    def __init__(self, kind: str, target: str, max_rows: int = 200, timeout: float = 30.0):
        self.kind = kind
        self.target = str(target)
        self.max_rows = max_rows
        self.timeout = timeout
        self._lock = threading.RLock()
        self._shared: Any = None  # cached in-memory DuckDB for files / postgres
        self._shared_signature: Any = None
        self._tmp: tempfile.TemporaryDirectory | None = None

    def _temp_directory(self) -> str:
        # DuckDB may read and write its temp folder. Use a private, empty one, so that
        # folder cannot be used to read other files (the default is ./.tmp or <db>.tmp).
        if self._tmp is None:
            self._tmp = tempfile.TemporaryDirectory(prefix="helm-duckdb-", ignore_cleanup_errors=True)
        return self._tmp.name

    @classmethod
    def from_settings(cls, settings: Settings) -> "Database":
        return cls(settings.db_kind, settings.db_target, settings.max_rows, settings.query_timeout)

    @property
    def dialect(self) -> str:
        return "sqlite" if self.kind == "sqlite" else "duckdb"

    # ------------------------------------------------------------------ connections

    def _missing(self) -> DatabaseError:
        return DatabaseError(
            f"Database not found: {self.target}. Run `uv run helm-demo` to create the demo data, "
            "or point HELM_DB (or helm.toml) at your own database."
        )

    @contextlib.contextmanager
    def _connect(self) -> Iterator[Any]:
        if self.kind == "duckdb":
            import duckdb

            if not Path(self.target).is_file():
                raise self._missing()
            try:
                con = duckdb.connect(
                    self.target,
                    read_only=True,
                    config={
                        "enable_external_access": False,
                        "lock_configuration": True,
                        "temp_directory": self._temp_directory(),
                    },
                )
            except duckdb.Error as exc:
                raise DatabaseError(f"Could not open DuckDB file {self.target}: {exc}") from None
            try:
                yield con
            finally:
                con.close()
        elif self.kind == "sqlite":
            path = Path(self.target)
            if not path.is_file():
                raise self._missing()
            uri = f"file:{pathname2url(str(path.resolve()))}?mode=ro"
            try:
                con = sqlite3.connect(uri, uri=True, check_same_thread=False)
                con.execute("PRAGMA query_only = ON")
            except sqlite3.Error as exc:
                raise DatabaseError(f"Could not open SQLite file {self.target}: {exc}") from None
            try:
                yield con
            finally:
                con.close()
        elif self.kind in ("files", "postgres"):
            with self._lock:
                yield self._shared_connection()
        else:
            raise DatabaseError(f"Unknown database kind: {self.kind}")

    def _data_files(self) -> list[Path]:
        root = Path(self.target)
        if root.is_file():
            return [root]
        if not root.is_dir():
            raise self._missing()
        files = []
        for path in sorted(root.rglob("*")):
            rel = path.relative_to(root)
            if any(part.startswith(".") for part in rel.parts):
                continue
            if path.is_file() and path.suffix.lower() in FILE_TABLE_SUFFIXES:
                files.append(path)
        return files

    def _shared_connection(self) -> Any:
        import duckdb

        if self.kind == "files":
            files = self._data_files()
            signature = tuple((str(p), p.stat().st_mtime_ns, p.stat().st_size) for p in files)
        else:
            signature = self.target
        if self._shared is not None and signature == self._shared_signature:
            return self._shared
        if self._shared is not None:
            with contextlib.suppress(Exception):
                self._shared.close()
            self._shared = None

        con = duckdb.connect(":memory:")
        try:
            if self.kind == "files":
                self._create_file_views(con, files)
            else:
                self._attach_postgres(con)
            con.execute(f"SET temp_directory = {_sql_text(self._temp_directory())}")
            con.execute("SET enable_external_access = false")
            con.execute("SET lock_configuration = true")
        except duckdb.Error as exc:
            con.close()
            raise DatabaseError(f"Could not load {self.kind} source {self.target}: {exc}") from None
        self._shared, self._shared_signature = con, signature
        return con

    def _create_file_views(self, con: Any, files: Sequence[Path]) -> None:
        root = Path(self.target)
        base = root if root.is_dir() else root.parent
        used: set[str] = set()
        for path in files:
            rel = path.relative_to(base).with_suffix("")
            name = re.sub(r"[^0-9a-zA-Z_]+", "_", rel.as_posix()).strip("_").lower() or "table"
            if name[0].isdigit():
                name = f"t_{name}"
            unique, n = name, 2
            while unique in used:
                unique, n = f"{name}_{n}", n + 1
            used.add(unique)
            literal = _sql_text(path.resolve())
            suffix = path.suffix.lower()
            if suffix == ".parquet":
                source = f"read_parquet({literal})"
            elif suffix == ".tsv":
                source = f"read_csv_auto({literal}, delim='\\t')"
            else:
                source = f"read_csv_auto({literal})"
            con.execute(f"CREATE VIEW {quote_ident(unique)} AS SELECT * FROM {source}")
        # Only the data files themselves stay readable, not the rest of the folder
        # (hidden files such as .env, or other files next to the data).
        allowed = ", ".join(_sql_text(path.resolve()) for path in files)
        con.execute(f"SET allowed_paths = [{allowed}]")

    def _attach_postgres(self, con: Any) -> None:
        # Experimental. Needs the DuckDB postgres extension (downloaded once by INSTALL).
        con.execute("INSTALL postgres")
        con.execute("LOAD postgres")
        dsn = self.target.replace("'", "''")
        con.execute(f"ATTACH '{dsn}' AS pg (TYPE postgres, READ_ONLY)")
        con.execute("USE pg")

    # ------------------------------------------------------------------ execution

    def _execute(
        self, con: Any, sql: str, params: Sequence[Any] = (), fetch: int | None = None, guarded: bool = False
    ) -> tuple[list[str], list[tuple[Any, ...]]]:
        stopped = threading.Event()

        def stop() -> None:
            stopped.set()
            with contextlib.suppress(Exception):
                con.interrupt()

        timer = threading.Timer(self.timeout, stop) if self.timeout else None
        if timer:
            timer.daemon = True
            timer.start()
        if guarded and self.kind == "sqlite":
            con.set_authorizer(_sqlite_authorizer)
        try:
            cursor = con.execute(sql, list(params))
            columns = [d[0] for d in (cursor.description or [])]
            rows = cursor.fetchmany(fetch) if fetch else cursor.fetchall()
            return columns, [tuple(r) for r in rows]
        except Exception as exc:
            if stopped.is_set():
                raise DatabaseError(
                    f"Query stopped after {self.timeout:g} seconds (time limit). "
                    "Add filters or aggregates to make it faster."
                ) from None
            message = str(exc).strip().splitlines()[0] if str(exc).strip() else type(exc).__name__
            if self.kind == "sqlite" and "not authorized" in message:
                message += " (only reading data is allowed)"
            raise DatabaseError(f"SQL error: {message}") from None
        finally:
            if timer:
                timer.cancel()
            if guarded and self.kind == "sqlite":
                con.set_authorizer(None)

    def _check_one_select(self, con: Any, sql: str) -> None:
        """A second opinion from DuckDB's own parser: the SQL must be one SELECT statement.

        The text guard already checks this. This check does not depend on how the guard
        reads quotes and comments, so a gap in the guard cannot run a second statement.
        """
        import duckdb

        try:
            statements = con.extract_statements(sql)
        except duckdb.Error as exc:
            message = str(exc).strip().splitlines()[0] if str(exc).strip() else type(exc).__name__
            raise DatabaseError(f"SQL error: {message}") from None
        # PIVOT without an IN list makes DuckDB add its own CREATE TYPE step, with no text.
        own = [s for s in statements if s.type != duckdb.StatementType.CREATE or s.query.strip()]
        if len(own) != 1 or own[0].type != duckdb.StatementType.SELECT:
            raise DatabaseError(
                "Blocked: only one read-only SELECT statement is allowed. This tool cannot change data or settings."
            )

    def query(self, sql: str, max_rows: int | None = None) -> QueryResult:
        """Run one read-only SELECT/WITH query with a row limit."""
        limit = max(1, min(int(max_rows or self.max_rows), self.max_rows))
        try:
            checked = check_sql(sql)
        except SQLGuardError as exc:
            raise DatabaseError(str(exc)) from None
        final = limit_rows(checked, limit)
        with self._connect() as con:
            if self.dialect == "duckdb":
                self._check_one_select(con, final)
            columns, rows = self._execute(con, final, fetch=limit + 1, guarded=True)
        truncated = len(rows) > limit
        return QueryResult(columns, rows[:limit], truncated, limit)

    # ------------------------------------------------------------------ schema

    def list_tables(self, with_counts: bool = True) -> list[TableInfo]:
        with self._connect() as con:
            if self.kind == "sqlite":
                _, rows = self._execute(
                    con,
                    "SELECT name, type FROM sqlite_master WHERE type IN ('table', 'view') "
                    "AND name NOT LIKE 'sqlite_%' ORDER BY name",
                )
                found = [("main", name, kind) for name, kind in rows]
            else:
                _, rows = self._execute(
                    con,
                    "SELECT table_schema, table_name, table_type FROM information_schema.tables "
                    "WHERE table_catalog = current_database() "
                    "AND table_schema NOT IN ('information_schema', 'pg_catalog') "
                    "ORDER BY table_schema, table_name",
                )
                found = [(s, t, "view" if "VIEW" in str(k).upper() else "table") for s, t, k in rows]

            tables = []
            for schema, table, kind in found:
                name = table if schema in _DEFAULT_SCHEMAS else f"{schema}.{table}"
                count = None
                if with_counts and self.kind != "postgres":
                    try:
                        _, result = self._execute(con, f"SELECT COUNT(*) FROM {self._qualified(schema, table)}")
                        count = int(result[0][0])
                    except DatabaseError as exc:
                        logger.warning("Could not count rows in %s: %s", name, exc)
                tables.append(TableInfo(name, schema, table, kind, count))
            return tables

    def _qualified(self, schema: str, table: str) -> str:
        if self.kind == "sqlite":
            return quote_ident(table)
        return f"{quote_ident(schema)}.{quote_ident(table)}"

    def find_table(self, name: str) -> TableInfo:
        wanted = (name or "").strip().strip('"').lower()
        tables = self.list_tables(with_counts=True)
        for info in tables:
            if wanted in (info.name.lower(), f"{info.schema}.{info.table}".lower()):
                return info
        known = ", ".join(t.name for t in tables) or "(none)"
        raise DatabaseError(f"Unknown table '{name}'. Known tables: {known}")

    def describe_table(self, name: str, sample_rows: int = 3) -> tuple[TableInfo, list[tuple[str, str]], QueryResult]:
        info = self.find_table(name)
        with self._connect() as con:
            if self.kind == "sqlite":
                _, rows = self._execute(con, f"PRAGMA table_info({quote_ident(info.table)})")
                columns = [(str(r[1]), str(r[2] or "")) for r in rows]
            else:
                _, rows = self._execute(
                    con,
                    "SELECT column_name, data_type FROM information_schema.columns "
                    "WHERE table_catalog = current_database() AND table_schema = ? AND table_name = ? "
                    "ORDER BY ordinal_position",
                    [info.schema, info.table],
                )
                columns = [(str(a), str(b)) for a, b in rows]
            cols, sample = self._execute(
                con, f"SELECT * FROM {self._qualified(info.schema, info.table)} LIMIT {int(sample_rows)}"
            )
        return info, columns, QueryResult(cols, sample, False, sample_rows)

    def close(self) -> None:
        with self._lock:
            if self._shared is not None:
                with contextlib.suppress(Exception):
                    self._shared.close()
                self._shared = None
            if self._tmp is not None:
                self._tmp.cleanup()
                self._tmp = None
