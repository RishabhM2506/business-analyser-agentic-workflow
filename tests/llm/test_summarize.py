"""Cassette-replay tests for the `summarize` node (docs/PLAN.md §7's
`llm`-marked layer). See `tests/llm/cassette.py`'s module docstring: every
cassette here is **synthetic** (no real Gemini key exists yet), not a real
recording — deterministic and free either way, which is the actual point
of this test layer (master brief §6).

Finding M11/QA-05: the CI `eval-gate` job's number-grounding check can
never organically fail, by construction — it always drives `summarize`
under `LLM_PROVIDER=mock`, and `MockLLM`'s own output is grounded by
construction (`app/models.py`'s `_mock_text_for` only ever echoes numbers
that are already present in its input). The fixture built to prove the
guardrail can actually catch a fabrication even via the cassette-replay
path — `tests/llm/cassettes/summarize/ungrounded_adversarial.json`, whose
own `_meta.note` explicitly says it exists for exactly this — was orphaned:
no test file used it. This file closes that gap; it is a `pytest.mark.llm`
test (cassette-replayed, zero token cost), a different CI job from
`eval-gate` itself, per QA-05's own concrete fix.
"""

from __future__ import annotations

import json

import pytest
from tests.llm.cassette import CASSETTES_DIR, CassetteModelClient

import app.nodes.summarize as summarize_module
from app.budget import BudgetTracker
from app.nodes.summarize import summarize
from app.schemas.errors import ErrorResponse
from app.schemas.query import TradeQuery
from app.schemas.response import CountryRow, TradeTable
from app.state import AnalysisState

_SUMMARIZE_CASSETTES = CASSETTES_DIR / "summarize"


def _table_normal() -> TradeTable:
    """Matches `cassettes/summarize/normal.json`'s hand-authored output —
    every number in that cassette's prose (6,350,000 / 2,110,000 /
    1,000,000 / 1,600,000 / 400,000 / 450,000 / 2019 / 2023 / the
    structural "5-year") must be a member of this table's flattened values
    for the grounded-replay test below to mean anything. Also reused (with
    zero changes) for the adversarial-cassette test: its fabricated figure,
    8,888,888, must NOT appear anywhere in this table."""
    return TradeTable(
        unit="USD",
        years=[2019, 2020, 2021, 2022, 2023],
        years_finalized=[2019, 2020, 2021, 2022, 2023],
        excluded_partner_codes=[],
        rows=[
            CountryRow(
                partner_country="United States",
                partner_code="842",
                values_by_year={
                    2019: 1_000_000.0,
                    2020: 1_250_000.0,
                    2021: 1_250_000.0,
                    2022: 1_250_000.0,
                    2023: 1_600_000.0,
                },
                cumulative_5yr=6_350_000.0,
                rank=1,
            ),
            CountryRow(
                partner_country="Germany",
                partner_code="276",
                values_by_year={
                    2019: 400_000.0,
                    2020: 420_000.0,
                    2021: 420_000.0,
                    2022: 420_000.0,
                    2023: 450_000.0,
                },
                cumulative_5yr=2_110_000.0,
                rank=2,
            ),
        ],
    )


def _table_provisional() -> TradeTable:
    """Matches `cassettes/summarize/provisional_year.json`'s hand-authored
    output: a single partner (China), a missing 2022 value, and 2023
    flagged as not yet finalized."""
    return TradeTable(
        unit="USD",
        years=[2019, 2020, 2021, 2022, 2023],
        years_finalized=[2019, 2020, 2021],
        excluded_partner_codes=[],
        rows=[
            CountryRow(
                partner_country="China",
                partner_code="156",
                values_by_year={
                    2019: 250_000.0,
                    2020: 250_000.0,
                    2021: 250_000.0,
                    2022: None,
                    2023: 350_000.0,
                },
                cumulative_5yr=1_100_000.0,
                rank=1,
            ),
        ],
    )


def _patch_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    # Fresh, generous-ceiling tracker per test (isolated from the
    # process-wide singleton) — these tests cover cassette-replay wiring
    # and guardrail integration, not budget enforcement.
    monkeypatch.setattr(
        summarize_module,
        "get_budget_tracker",
        lambda: BudgetTracker(max_calls_per_thread=100, max_calls_per_day=100),
    )


@pytest.mark.llm
async def test_summarize_replays_grounded_cassette(monkeypatch: pytest.MonkeyPatch) -> None:
    cassette = CassetteModelClient(_SUMMARIZE_CASSETTES / "normal.json")
    monkeypatch.setattr(summarize_module, "get_model_for_role", lambda role, provider: cassette)
    _patch_budget(monkeypatch)
    table = _table_normal()
    state: AnalysisState = {
        "query": TradeQuery(hs_code="010121"),
        "imports_table": table,
        "exports_table": table,
        "thread_id": "llm-summarize-normal",
    }

    result = await summarize(state)

    assert "analytical_summary" in result
    assert "error" not in result


@pytest.mark.llm
async def test_summarize_replays_provisional_year_cassette(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cassette = CassetteModelClient(_SUMMARIZE_CASSETTES / "provisional_year.json")
    monkeypatch.setattr(summarize_module, "get_model_for_role", lambda role, provider: cassette)
    _patch_budget(monkeypatch)
    table = _table_provisional()
    state: AnalysisState = {
        "query": TradeQuery(hs_code="010121"),
        "imports_table": table,
        "exports_table": table,
        "thread_id": "llm-summarize-provisional",
    }

    result = await summarize(state)

    assert "analytical_summary" in result
    assert "error" not in result


@pytest.mark.llm
async def test_summarize_replays_ungrounded_adversarial_cassette_and_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """M11/QA-05: the exact gap the orphaned `ungrounded_adversarial.json`
    fixture was built to close. Proves the output guardrail catches a
    fabricated number even when it arrives via the cassette-replay path —
    not just via a hand-rolled test double (`tests/integration/test_summarize.py`
    already covers that) — and that a real, CI-runnable test exercises it."""
    cassette = CassetteModelClient(_SUMMARIZE_CASSETTES / "ungrounded_adversarial.json")
    monkeypatch.setattr(summarize_module, "get_model_for_role", lambda role, provider: cassette)
    _patch_budget(monkeypatch)
    table = _table_normal()  # 8,888,888 (the cassette's fabricated figure) appears nowhere in it
    state: AnalysisState = {
        "query": TradeQuery(hs_code="010121"),
        "imports_table": table,
        "exports_table": table,
        "thread_id": "llm-summarize-adversarial",
        "trace_id": "t-adversarial",
    }

    result = await summarize(state)

    assert "analytical_summary" not in result
    error = result["error"]
    assert isinstance(error, ErrorResponse)
    assert error.error_code == "UNGROUNDED_SUMMARY"
    assert error.trace_id == "t-adversarial"


@pytest.mark.llm
async def test_summarize_cassette_outputs_are_schema_valid_and_marked_synthetic() -> None:
    for cassette_path in _SUMMARIZE_CASSETTES.glob("*.json"):
        payload = json.loads(cassette_path.read_text(encoding="utf-8"))
        assert payload["_meta"]["synthetic"] is True, f"{cassette_path} must be marked synthetic"
        assert "analytical_summary" in payload["output"]
        assert len(payload["output"]["analytical_summary"]) > 0
