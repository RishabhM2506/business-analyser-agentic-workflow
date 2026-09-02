"""Unit tests for `app.report.narrative` — grounding-set generalization,
prompt rendering, the deterministic template fallback (grounded by
construction), and the reject -> regenerate once -> template-fallback
orchestration using a fake `ModelClient` (mirrors
`tests/integration/test_describe_item.py`'s `_NumberInventingModelClient`
pattern — no DB, no network needed for any of this).
"""

from __future__ import annotations

from datetime import date
from typing import TypeVar

import pytest
from pydantic import BaseModel

from app.budget import BudgetExceededError, BudgetTracker
from app.report.facts import (
    AllOtherPartnersFact,
    AnnualSeriesYear,
    Facts,
    HhiYear,
    InternationalProductionFact,
    MandiPriceFact,
    MspFact,
    PartnerFact,
    UnitValueTrendYear,
    Window,
)
from app.report.narrative import (
    NarrativeOutput,
    _load_system_prompt,
    check_narrative_grounded,
    find_ungrounded_numbers,
    flatten_facts_numbers,
    generate_narrative,
    render_facts_for_prompt,
    render_template_fallback,
)

pytestmark = pytest.mark.unit

T = TypeVar("T", bound=BaseModel)


def _facts(*, regulatory_note_missing_warning: bool = False) -> Facts:
    return Facts(
        hs6="120791",
        product_label="Oil seeds; poppy seeds, whether or not broken",
        flow="import",
        window=Window(years=2, start_year=2022, end_year=2023),
        top_n=10,
        annual_series=[
            AnnualSeriesYear(
                year=2022,
                flow="import",
                total_inr_paise=424_660_000_000,
                status="QTY_MISSING",
                partners=[
                    PartnerFact(
                        rank=1,
                        country="Türkiye",
                        partner_country_code="792",
                        value_inr_paise=424_660_000_000,
                        status="QTY_MISSING",
                    )
                ],
                all_other_partners=AllOtherPartnersFact(value_inr_paise=0, status="OK"),
            ),
            AnnualSeriesYear(
                year=2023,
                flow="import",
                total_inr_paise=0,
                status="ZERO",
                partners=[
                    PartnerFact(
                        rank=1,
                        country="Türkiye",
                        partner_country_code="792",
                        value_inr_paise=0,
                        status="ZERO",
                    )
                ],
                all_other_partners=AllOtherPartnersFact(value_inr_paise=0, status="OK"),
            ),
        ],
        month_wise_current_year=[],
        unit_value_trend=[
            UnitValueTrendYear(
                year=2022,
                inr_paise_per_kg=25800,
                delta_qty_pct=None,
                delta_price_pct=None,
                delta_fx_pct=None,
            ),
            UnitValueTrendYear(
                year=2023,
                inr_paise_per_kg=None,
                delta_qty_pct=None,
                delta_price_pct=None,
                delta_fx_pct=None,
            ),
        ],
        hhi_by_year=[HhiYear(year=2022, hhi=1), HhiYear(year=2023, hhi=None)],
        overall_cagr=None,
        overall_volatility=None,
        cagr_by_partner={},
        volatility_by_partner={},
        landed_cost=None,
        landed_cost_as_of_period=None,
        mismatch_checks=[],
        regulatory_note=None,
        regulatory_note_missing_warning=regulatory_note_missing_warning,
        coverage=None,
        hs8_split_note="12079100 is the only ITC-HS8 line beneath 120791 as of this vintage.",
        mandi_price=MandiPriceFact(
            status="NOT_APPLICABLE",
            matched_commodity=None,
            modal_price_inr_paise_per_qtl=None,
            price_date=None,
            market=None,
            state=None,
        ),
        msp=MspFact(
            status="NOT_APPLICABLE",
            matched_commodity=None,
            year_label=None,
            msp_inr_paise_per_qtl=None,
            cost_inr_paise_per_qtl=None,
        ),
        international_production=InternationalProductionFact(
            status="NOT_APPLICABLE",
            matched_item=None,
            year=None,
            india_status=None,
            india_production_tonnes=None,
            world_production_tonnes=None,
        ),
        llm_datapoints=[],
        mandi_price_llm_datapoints=[],
        msp_llm_datapoints=[],
        international_production_llm_datapoints=[],
    )


class _FixedModelClient:
    def __init__(self, texts: list[str]) -> None:
        self._texts = list(texts)

    async def generate_structured(
        self, *, system_prompt: str, user_content: str, schema: type[T]
    ) -> T:
        return schema.model_validate({"narrative": self._texts.pop(0)})


def _budget_tracker(*, max_calls_per_thread: int = 100) -> BudgetTracker:
    return BudgetTracker(max_calls_per_thread=max_calls_per_thread, max_calls_per_day=100)


def test_flatten_facts_numbers_includes_both_paise_and_crore() -> None:
    grounded = flatten_facts_numbers(_facts())

    assert 424_660_000_000.0 in grounded  # raw paise
    assert 424.66 in grounded  # crore conversion (424_660_000_000 paise = 424.66 crore)
    assert 2022.0 in grounded  # a plain structural year is also grounded
    assert 1.0 in grounded  # rank 1 / HHI 1 are also grounded


def test_llm_datapoints_are_never_rendered_into_the_model_prompt() -> None:
    """2026-09-02, Step 4 hardening: the model must never see a cited-but-
    not-independently-verified figure, so it can never narrate one. This
    is the actual protection mechanism - not a number-matching filter."""
    from app.report.facts import LlmDatapointFact

    facts = _facts().model_copy(
        update={
            "llm_datapoints": [
                LlmDatapointFact(
                    field_name="mandi_price",
                    effective_period="2026-08",
                    value={"modal_price_inr_paise_per_qtl": 999_999},
                    source_authority="Test Authority",
                    source_reference="Test",
                    source_url=None,
                    verified_date=date(2026, 9, 2),
                )
            ],
            "mandi_price_llm_datapoints": [],
        }
    )
    assert "999999" not in render_facts_for_prompt(facts)
    assert "999,999" not in render_facts_for_prompt(facts)


def test_llm_datapoints_numbers_are_technically_groundable_but_never_reach_the_prompt() -> None:
    """Documents a known, deliberate tradeoff (see this module's own
    docstring): `flatten_facts_numbers` is fully generic over `Facts`'
    shape, so an `llm_datapoints` number is technically part of the
    groundable set - this is harmless only because the number is never
    rendered into the prompt in the first place (previous test), so a
    model has no way to guess it by coincidence."""
    from app.report.facts import LlmDatapointFact

    facts = _facts().model_copy(
        update={
            "llm_datapoints": [
                LlmDatapointFact(
                    field_name="mandi_price",
                    effective_period="2026-08",
                    value={"modal_price_inr_paise_per_qtl": 777_777},
                    source_authority="Test Authority",
                    source_reference="Test",
                    source_url=None,
                    verified_date=date(2026, 9, 2),
                )
            ]
        }
    )
    assert 777777.0 in flatten_facts_numbers(facts)


def test_check_narrative_grounded_accepts_a_real_grounded_number() -> None:
    assert check_narrative_grounded("In 2022, imports totalled ₹424.66 crore.", _facts())


def test_check_narrative_grounded_rejects_an_invented_number() -> None:
    assert not check_narrative_grounded("Imports grew by 37% that year.", _facts())


def test_find_ungrounded_numbers_reports_exactly_the_invented_one() -> None:
    ungrounded = find_ungrounded_numbers(
        "In 2022, imports totalled ₹424.66 crore, up 37% year on year.", _facts()
    )
    assert ungrounded == [37.0]


def test_render_facts_for_prompt_contains_the_real_figures() -> None:
    rendered = render_facts_for_prompt(_facts())

    assert "120791" in rendered
    assert "2022" in rendered
    assert "Türkiye" in rendered
    assert "QTY_MISSING" in rendered
    assert "ZERO" in rendered


def test_render_template_fallback_is_grounded_by_construction() -> None:
    facts = _facts()
    fallback = render_template_fallback(facts)

    assert check_narrative_grounded(fallback, facts)


def test_render_template_fallback_never_states_a_zero_year_as_missing() -> None:
    fallback = render_template_fallback(_facts())

    assert "2023" in fallback
    assert "ZERO" in fallback


def test_load_system_prompt_omits_concentration_clause_by_default() -> None:
    """ "concentration" alone isn't a safe marker - the base prompt's own
    general description legitimately mentions "partner-concentration
    (HHI)"; "commercial-preference" only ever appears in the conditional
    clause."""
    prompt = _load_system_prompt(regulatory_note_missing_warning=False)

    assert "commercial-preference" not in prompt.lower()


def test_load_system_prompt_includes_concentration_clause_when_flagged() -> None:
    prompt = _load_system_prompt(regulatory_note_missing_warning=True)

    assert "concentration" in prompt.lower()
    assert "commercial-preference" in prompt.lower()


async def test_generate_narrative_accepts_a_grounded_first_attempt() -> None:
    facts = _facts()
    client = _FixedModelClient(["In 2022, imports totalled ₹424.66 crore."])

    result = await generate_narrative(
        facts,
        model_client=client,
        budget_tracker=_budget_tracker(),
        thread_id="t-1",
        tenant_id="default",
    )

    assert result.source == "model"
    assert "424.66" in result.narrative


async def test_generate_narrative_retries_once_then_accepts() -> None:
    facts = _facts()
    client = _FixedModelClient(
        [
            "Imports grew by 37% that year.",  # ungrounded - rejected
            "In 2022, imports totalled ₹424.66 crore.",  # grounded - accepted
        ]
    )

    result = await generate_narrative(
        facts,
        model_client=client,
        budget_tracker=_budget_tracker(),
        thread_id="t-1",
        tenant_id="default",
    )

    assert result.source == "model_retry"


async def test_generate_narrative_falls_back_to_template_after_two_ungrounded_attempts() -> None:
    facts = _facts()
    client = _FixedModelClient(
        ["Imports grew by 37% that year.", "Exports fell by 12% year on year."]
    )

    result = await generate_narrative(
        facts,
        model_client=client,
        budget_tracker=_budget_tracker(),
        thread_id="t-1",
        tenant_id="default",
    )

    assert result.source == "template_fallback"
    assert check_narrative_grounded(result.narrative, facts)


async def test_generate_narrative_propagates_budget_exceeded_rather_than_degrading() -> None:
    """Matches app.search.service.search_products's own precedent: budget
    exhaustion is a distinct, real failure the caller (the route) tells the
    user about via BUDGET_EXCEEDED - it must never be silently swallowed
    into a degraded-quality template narrative, which exists only for "the
    model tried and failed grounding," a different real failure mode."""
    facts = _facts()
    client = _FixedModelClient(["Imports grew by 37% that year."])  # never reached

    with pytest.raises(BudgetExceededError):
        await generate_narrative(
            facts,
            model_client=client,
            budget_tracker=_budget_tracker(max_calls_per_thread=0),
            thread_id="t-1",
            tenant_id="default",
        )


def test_narrative_output_schema_rejects_empty_string() -> None:
    with pytest.raises(ValueError):
        NarrativeOutput.model_validate({"narrative": ""})
