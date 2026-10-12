import asyncio
from pathlib import Path

import pytest

from peakwhale_helm.assistant import Assistant
from peakwhale_helm.evals import check_case, load_cases, main, run_cases
from peakwhale_helm.sources import Source
from peakwhale_helm.toolbox import Toolbox
from peakwhale_helm.toolset import LocalToolSet

REPO = Path(__file__).resolve().parents[1]


def test_all_demo_cases_pass_offline_with_the_fallback(demo_settings):
    cases = load_cases(REPO / "evals" / "demo_cases.json")
    assert len(cases) >= 10
    assistant = Assistant(LocalToolSet(Toolbox(demo_settings)), demo_settings)
    results = asyncio.run(run_cases(assistant, cases))
    failed = {r.id: r.problems for r in results if not r.passed}
    assert failed == {}


def test_check_case_rules():
    case = {"expect": ["8", ["95,000", "95000"]], "tools": ["query"]}
    good = "Customer has 8 items and earns 95000.\n\n**Sources**\n- [S1] query"
    assert check_case(case, good, [Source("S1", "query")]) == []
    # "8" inside "2631d00b8" or only in the Sources list does not count
    bad = "Id 2631d008b.\n\n**Sources**\n- 8 95,000"
    problems = check_case(case, bad, [])
    assert problems == ['missing "8"', 'missing "95,000" or "95000"', "tool query not used"]


def test_helm_eval_command_with_fresh_demo(monkeypatch, tmp_path):
    monkeypatch.chdir(REPO)
    with pytest.raises(SystemExit) as info:
        main(["--fresh-demo"])
    assert info.value.code == 0
