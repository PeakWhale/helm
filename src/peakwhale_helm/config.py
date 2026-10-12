"""Settings for Helm.

Settings come from three places, in this order (later wins):

1. Built-in defaults: the bundled synthetic demo (``data/ledger.duckdb`` and ``data/vault``).
2. A TOML file: ``HELM_CONFIG`` if set, else ``./helm.toml`` if it exists.
3. Environment variables such as ``HELM_DB`` and ``HELM_DOCS``.

Relative paths in the TOML file are resolved from the folder of that file.
Relative paths in environment variables and defaults are resolved from the
current working directory.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping

DEFAULT_DB = "data/ledger.duckdb"
DEFAULT_DOCS = "data/vault"

DB_KINDS = ("duckdb", "sqlite", "files", "postgres")
FILE_TABLE_SUFFIXES = (".csv", ".tsv", ".parquet")
RETRIEVERS = ("bm25", "embeddings")
LLM_PROVIDERS = ("ollama", "openai", "none")


class ConfigError(ValueError):
    """Raised when the configuration is not valid."""


@dataclass(frozen=True)
class SavedQuery:
    """A named, reviewed SQL query. The no-LLM fallback runs it when a keyword matches."""

    name: str
    sql: str
    description: str = ""
    keywords: tuple[str, ...] = ()


@dataclass(frozen=True)
class LLMSettings:
    provider: str = "ollama"  # ollama | openai | none
    model: str | None = None
    base_url: str | None = None
    api_key: str | None = None
    temperature: float = 0.0
    max_steps: int = 8

    @property
    def model_name(self) -> str | None:
        if self.model:
            return self.model
        if self.provider == "ollama":
            return "llama3.2"
        return None


@dataclass(frozen=True)
class Settings:
    db_target: str = DEFAULT_DB
    db_kind: str = "duckdb"
    docs_dir: Path | None = Path(DEFAULT_DOCS)
    max_rows: int = 200
    query_timeout: float = 30.0
    retriever: str = "bm25"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_size: int = 800
    chunk_overlap: int = 120
    enable_market: bool = False
    enable_sentiment: bool = False
    saved_queries: tuple[SavedQuery, ...] = ()
    llm: LLMSettings = field(default_factory=LLMSettings)
    config_file: Path | None = None

    def with_data(self, db: str | Path, docs: str | Path | None) -> "Settings":
        """Return a copy that points at another database and documents folder."""
        target = str(db)
        return replace(
            self,
            db_target=target,
            db_kind=detect_db_kind(target),
            docs_dir=Path(docs) if docs is not None else None,
        )


_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off", ""}


def _parse_bool(value: Any, name: str) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in _TRUE:
        return True
    if text in _FALSE:
        return False
    raise ConfigError(f"{name} must be true or false, got {value!r}")


def _parse_int(value: Any, name: str, minimum: int = 1) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        raise ConfigError(f"{name} must be a whole number, got {value!r}") from None
    if number < minimum:
        raise ConfigError(f"{name} must be at least {minimum}, got {number}")
    return number


def _parse_float(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ConfigError(f"{name} must be a number, got {value!r}") from None
    if number < 0:
        raise ConfigError(f"{name} must not be negative, got {number}")
    return number


def detect_db_kind(target: str) -> str:
    """Guess the database kind from a path or URL."""
    text = str(target).strip()
    lower = text.lower()
    if lower.startswith(("postgres://", "postgresql://")):
        return "postgres"
    path = Path(text)
    if path.is_dir() or lower.endswith(FILE_TABLE_SUFFIXES):
        return "files"
    if path.is_file():
        try:
            with path.open("rb") as handle:
                head = handle.read(16)
        except OSError:
            head = b""
        if head.startswith(b"SQLite format 3\x00"):
            return "sqlite"
        if head[8:12] == b"DUCK":
            return "duckdb"
    if lower.endswith((".sqlite", ".sqlite3", ".db3")):
        return "sqlite"
    return "duckdb"


def _resolve(value: str, base: Path) -> str:
    if value.lower().startswith(("postgres://", "postgresql://")):
        return value
    path = Path(os.path.expanduser(value))
    if not path.is_absolute():
        path = base / path
    return str(path)


def _find_config_file(explicit: str | Path | None, env: Mapping[str, str]) -> Path | None:
    if explicit:
        path = Path(explicit).expanduser()
        if not path.is_file():
            raise ConfigError(f"Config file not found: {path}")
        return path.resolve()
    env_path = env.get("HELM_CONFIG", "").strip()
    if env_path:
        path = Path(env_path).expanduser()
        if not path.is_file():
            raise ConfigError(f"HELM_CONFIG points to a missing file: {path}")
        return path.resolve()
    local = Path.cwd() / "helm.toml"
    return local.resolve() if local.is_file() else None


def _saved_queries(items: Any) -> tuple[SavedQuery, ...]:
    if not items:
        return ()
    if not isinstance(items, list):
        raise ConfigError("saved_queries must be a list of [[saved_queries]] tables")
    result = []
    for i, item in enumerate(items):
        if not isinstance(item, dict) or not item.get("name") or not item.get("sql"):
            raise ConfigError(f"saved_queries entry {i + 1} needs a name and sql")
        keywords = item.get("keywords", [])
        if isinstance(keywords, str):
            keywords = [keywords]
        result.append(
            SavedQuery(
                name=str(item["name"]),
                sql=str(item["sql"]).strip(),
                description=str(item.get("description", "")),
                keywords=tuple(str(k) for k in keywords),
            )
        )
    return tuple(result)


def load_settings(
    config_path: str | Path | None = None,
    env: Mapping[str, str] | None = None,
) -> Settings:
    """Load settings from defaults, an optional TOML file and environment variables."""
    env = os.environ if env is None else env
    cwd = Path.cwd()

    values: dict[str, Any] = {}
    db_kind: str | None = None
    db_target = _resolve(DEFAULT_DB, cwd)
    docs: str | None = _resolve(DEFAULT_DOCS, cwd)
    llm: dict[str, Any] = {}
    saved: tuple[SavedQuery, ...] = ()

    config_file = _find_config_file(config_path, env)
    if config_file:
        try:
            data = tomllib.loads(config_file.read_text(encoding="utf-8"))
        except tomllib.TOMLDecodeError as exc:
            raise ConfigError(f"{config_file}: {exc}") from None
        base = config_file.parent

        database = data.get("database", {})
        if database.get("url"):
            db_target = str(database["url"])
        elif database.get("path"):
            db_target = _resolve(str(database["path"]), base)
        if database.get("kind"):
            db_kind = str(database["kind"]).lower()

        documents = data.get("documents", {})
        if "path" in documents:
            docs = _resolve(str(documents["path"]), base) if documents["path"] else None
        for key, name in (("retriever", "retriever"), ("embedding_model", "embedding_model")):
            if key in documents:
                values[name] = str(documents[key])
        if "chunk_size" in documents:
            values["chunk_size"] = _parse_int(documents["chunk_size"], "documents.chunk_size", 100)
        if "chunk_overlap" in documents:
            values["chunk_overlap"] = _parse_int(documents["chunk_overlap"], "documents.chunk_overlap", 0)

        limits = data.get("limits", {})
        if "max_rows" in limits:
            values["max_rows"] = _parse_int(limits["max_rows"], "limits.max_rows")
        if "query_timeout_seconds" in limits:
            values["query_timeout"] = _parse_float(limits["query_timeout_seconds"], "limits.query_timeout_seconds")

        tools = data.get("tools", {})
        if "market" in tools:
            values["enable_market"] = _parse_bool(tools["market"], "tools.market")
        if "sentiment" in tools:
            values["enable_sentiment"] = _parse_bool(tools["sentiment"], "tools.sentiment")

        llm.update({k: v for k, v in data.get("llm", {}).items()})
        saved = _saved_queries(data.get("saved_queries"))

    # Environment variables win over the file.
    env_db = env.get("HELM_DB", "").strip()
    if env_db:
        new_target = _resolve(env_db, cwd)
        if saved and new_target != db_target:
            # Saved queries are written for one database. Do not run them on another one.
            saved = ()
        db_target = new_target
    if env.get("HELM_DB_KIND", "").strip():
        db_kind = env["HELM_DB_KIND"].strip().lower()
    if "HELM_DOCS" in env:
        docs = _resolve(env["HELM_DOCS"].strip(), cwd) if env["HELM_DOCS"].strip() else None
    if env.get("HELM_MAX_ROWS"):
        values["max_rows"] = _parse_int(env["HELM_MAX_ROWS"], "HELM_MAX_ROWS")
    if env.get("HELM_QUERY_TIMEOUT"):
        values["query_timeout"] = _parse_float(env["HELM_QUERY_TIMEOUT"], "HELM_QUERY_TIMEOUT")
    if env.get("HELM_RETRIEVER"):
        values["retriever"] = env["HELM_RETRIEVER"].strip().lower()
    if env.get("HELM_EMBEDDING_MODEL"):
        values["embedding_model"] = env["HELM_EMBEDDING_MODEL"].strip()
    if "HELM_ENABLE_MARKET" in env:
        values["enable_market"] = _parse_bool(env["HELM_ENABLE_MARKET"], "HELM_ENABLE_MARKET")
    if "HELM_ENABLE_SENTIMENT" in env:
        values["enable_sentiment"] = _parse_bool(env["HELM_ENABLE_SENTIMENT"], "HELM_ENABLE_SENTIMENT")

    # LLM settings. Older HELM_OLLAMA_* names still work.
    if "HELM_USE_OLLAMA" in env and not _parse_bool(env["HELM_USE_OLLAMA"], "HELM_USE_OLLAMA"):
        llm["provider"] = "none"
    if env.get("HELM_OLLAMA_MODEL"):
        llm["model"] = env["HELM_OLLAMA_MODEL"].strip()
    if env.get("HELM_OLLAMA_BASE_URL"):
        llm["base_url"] = env["HELM_OLLAMA_BASE_URL"].strip()
    for key in ("provider", "model", "base_url", "api_key", "temperature"):
        env_value = env.get(f"HELM_LLM_{key.upper()}", "").strip()
        if env_value:
            llm[key] = env_value
    if env.get("HELM_MAX_STEPS"):
        llm["max_steps"] = env["HELM_MAX_STEPS"]

    provider = str(llm.get("provider", "ollama")).strip().lower()
    if provider not in LLM_PROVIDERS:
        raise ConfigError(f"LLM provider must be one of {', '.join(LLM_PROVIDERS)}, got {provider!r}")
    api_key = llm.get("api_key") or (env.get("OPENAI_API_KEY") if provider == "openai" else None)
    llm_settings = LLMSettings(
        provider=provider,
        model=str(llm["model"]) if llm.get("model") else None,
        base_url=str(llm["base_url"]) if llm.get("base_url") else None,
        api_key=str(api_key) if api_key else None,
        temperature=_parse_float(llm.get("temperature", 0.0), "llm.temperature"),
        max_steps=_parse_int(llm.get("max_steps", 8), "llm.max_steps"),
    )

    kind = db_kind or detect_db_kind(db_target)
    if kind not in DB_KINDS:
        raise ConfigError(f"Database kind must be one of {', '.join(DB_KINDS)}, got {kind!r}")
    retriever = values.get("retriever", "bm25")
    if retriever not in RETRIEVERS:
        raise ConfigError(f"Retriever must be one of {', '.join(RETRIEVERS)}, got {retriever!r}")

    return Settings(
        db_target=db_target,
        db_kind=kind,
        docs_dir=Path(docs) if docs else None,
        saved_queries=saved,
        llm=llm_settings,
        config_file=config_file,
        **values,
    )


def describe_settings(settings: Settings) -> str:
    """A short, human readable summary (no secrets)."""
    lines = [
        f"Config file : {settings.config_file or '(none, using defaults and environment)'}",
        f"Database    : {settings.db_kind} -> {redact_url(settings.db_target)}",
        f"Documents   : {settings.docs_dir or '(none)'} (retriever: {settings.retriever})",
        f"Limits      : {settings.max_rows} rows, {settings.query_timeout:g} s per query",
        f"Extra tools : market={'on' if settings.enable_market else 'off'}, "
        f"sentiment={'on' if settings.enable_sentiment else 'off'}",
        f"LLM         : {settings.llm.provider}"
        + (f" ({settings.llm.model_name})" if settings.llm.provider != "none" else ""),
    ]
    if settings.saved_queries:
        lines.append("Saved SQL   : " + ", ".join(q.name for q in settings.saved_queries))
    return "\n".join(lines)


def redact_url(target: str) -> str:
    """Hide a password inside a database URL."""
    if "://" in target and "@" in target:
        scheme, rest = target.split("://", 1)
        creds, host = rest.rsplit("@", 1)
        user = creds.split(":", 1)[0]
        return f"{scheme}://{user}:***@{host}"
    return target
