import sqlite3
from pathlib import Path

import duckdb
import pytest

from peakwhale_helm.config import ConfigError, describe_settings, detect_db_kind, load_settings, redact_url
from peakwhale_helm.demo import DEMO_SAVED_QUERIES

REPO = Path(__file__).resolve().parents[1]


def test_defaults_are_the_demo(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    s = load_settings(env={})
    assert s.db_target == str(tmp_path / "data" / "ledger.duckdb")
    assert s.docs_dir == tmp_path / "data" / "vault"
    assert s.db_kind == "duckdb"
    assert s.llm.provider == "ollama" and s.llm.model_name == "llama3.2"
    assert not s.enable_market and not s.enable_sentiment
    assert s.config_file is None


def test_repo_helm_toml_matches_demo_saved_queries(monkeypatch):
    monkeypatch.chdir(REPO)
    s = load_settings(env={})
    assert s.config_file == REPO / "helm.toml"
    assert s.saved_queries == DEMO_SAVED_QUERIES
    assert s.db_target == str(REPO / "data" / "ledger.duckdb")


def test_toml_file_and_relative_paths(tmp_path, monkeypatch):
    cfg_dir = tmp_path / "cfg"
    cfg_dir.mkdir()
    (cfg_dir / "my.toml").write_text(
        """
[database]
path = "warehouse"
[documents]
path = "docs"
retriever = "bm25"
[limits]
max_rows = 50
query_timeout_seconds = 5
[tools]
market = true
[llm]
provider = "openai"
model = "some-model"
base_url = "http://localhost:1234/v1"
[[saved_queries]]
name = "q"
sql = "SELECT 1"
keywords = "one"
"""
    )
    (cfg_dir / "warehouse").mkdir()
    monkeypatch.chdir(tmp_path)
    s = load_settings(env={"HELM_CONFIG": "cfg/my.toml"})
    assert s.db_target == str(cfg_dir / "warehouse")
    assert s.db_kind == "files"
    assert s.docs_dir == cfg_dir / "docs"
    assert s.max_rows == 50 and s.query_timeout == 5
    assert s.enable_market is True
    assert s.llm.provider == "openai" and s.llm.model_name == "some-model"
    assert s.saved_queries[0].keywords == ("one",)


def test_env_wins_and_drops_saved_queries_for_another_db(tmp_path, monkeypatch):
    monkeypatch.chdir(REPO)
    s = load_settings(
        env={
            "HELM_DB": str(tmp_path / "other.duckdb"),
            "HELM_DOCS": "",
            "HELM_MAX_ROWS": "10",
            "HELM_ENABLE_SENTIMENT": "yes",
            "HELM_LLM_PROVIDER": "none",
        }
    )
    assert s.db_target == str(tmp_path / "other.duckdb")
    assert s.saved_queries == ()
    assert s.docs_dir is None
    assert s.max_rows == 10
    assert s.enable_sentiment is True
    assert s.llm.provider == "none"


def test_old_ollama_variables_still_work(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    s = load_settings(env={"HELM_OLLAMA_MODEL": "qwen2.5", "HELM_OLLAMA_BASE_URL": "http://box:11434"})
    assert s.llm.model_name == "qwen2.5" and s.llm.base_url == "http://box:11434"
    assert load_settings(env={"HELM_USE_OLLAMA": "0"}).llm.provider == "none"


def test_openai_key_from_standard_variable(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    s = load_settings(env={"HELM_LLM_PROVIDER": "openai", "OPENAI_API_KEY": "sk-test"})
    assert s.llm.api_key == "sk-test"
    assert "sk-test" not in describe_settings(s)


@pytest.mark.parametrize(
    "env, message",
    [
        ({"HELM_MAX_ROWS": "zero"}, "whole number"),
        ({"HELM_MAX_ROWS": "0"}, "at least 1"),
        ({"HELM_ENABLE_MARKET": "maybe"}, "true or false"),
        ({"HELM_LLM_PROVIDER": "skynet"}, "provider"),
        ({"HELM_DB_KIND": "oracle"}, "Database kind"),
        ({"HELM_RETRIEVER": "magic"}, "Retriever"),
        ({"HELM_CONFIG": "missing.toml"}, "missing file"),
    ],
)
def test_bad_values_give_clear_errors(env, message, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ConfigError, match=message):
        load_settings(env=env)


def test_detect_db_kind(tmp_path):
    sqlite_path = tmp_path / "shop.db"
    sqlite3.connect(sqlite_path).execute("CREATE TABLE t (x)").connection.commit()
    duck_path = tmp_path / "shop2.db"
    duckdb.connect(str(duck_path)).execute("CREATE TABLE t (x INT)").close()
    assert detect_db_kind(str(sqlite_path)) == "sqlite"
    assert detect_db_kind(str(duck_path)) == "duckdb"
    assert detect_db_kind(str(tmp_path)) == "files"
    assert detect_db_kind("sales.parquet") == "files"
    assert detect_db_kind("new.sqlite") == "sqlite"
    assert detect_db_kind("postgresql://me@host/db") == "postgres"


def test_password_is_hidden_in_summary(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    s = load_settings(env={"HELM_DB": "postgresql://me:secret@host:5432/db"})
    text = describe_settings(s)
    assert "secret" not in text and "me:***@host:5432/db" in text


def test_password_is_hidden_in_other_connection_strings():
    assert redact_url("host=db user=me password=secret dbname=shop") == "host=db user=me password=*** dbname=shop"
    assert redact_url("postgresql://me@host/db?password=secret&sslmode=require") == (
        "postgresql://me:***@host/db?password=***&sslmode=require"
    )
