"""Shared fixtures. Everything runs offline: no LLM, no network, no model downloads."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from peakwhale_helm.config import Settings, load_settings
from peakwhale_helm.demo import DEMO_SAVED_QUERIES, generate


@pytest.fixture(scope="session")
def demo_dir(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("demo")
    generate(out)
    return out


@pytest.fixture()
def demo_settings(demo_dir, monkeypatch, tmp_path) -> Settings:
    # Run from an empty folder so the repo's helm.toml is not picked up.
    monkeypatch.chdir(tmp_path)
    settings = load_settings(env={"HELM_DB": str(demo_dir / "ledger.duckdb"), "HELM_DOCS": str(demo_dir / "vault")})
    return replace(settings, saved_queries=DEMO_SAVED_QUERIES)


@pytest.fixture()
def toolbox(demo_settings):
    from peakwhale_helm.toolbox import Toolbox

    tb = Toolbox(demo_settings)
    yield tb
    tb.close()
