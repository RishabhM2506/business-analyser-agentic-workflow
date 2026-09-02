"""Trade-report narrative: MODEL_ANALYSIS call over `app.report.facts.Facts`
only — never raw warehouse tables (mirrors `app/nodes/summarize.py`'s
"pre-aggregated table only" discipline, docs/PLAN.md §2.2/§14). Output must
pass `check_narrative_grounded` before being returned; unlike
`summarize.py`'s single-shot reject, §14's own spec for this flow is
"reject -> regenerate once -> template fallback" — the template fallback
(`render_template_fallback`) is a fully deterministic, code-generated
narrative built directly from `Facts`, guaranteed grounded by construction,
so a caller always gets *some* narrative back, never a bare error.

**Grounding set generalizes `app.guardrails.check_numbers_grounded`'s
pattern** rather than reusing it directly — that function is hardwired to
`TradeTable`'s shape. Every numeric leaf of `Facts` (including structural
ones like `year`/`rank`/`top_n`) is grounded, which is actually simpler
than `guardrails.py`'s original design: that module needed a separate
structural-number allowlist because `rank`/`count` weren't literally
"value" fields in `TradeTable`'s own JSON; every one of `Facts`' fields
*is* a real leaf of the document being grounded against, so no such
allowlist is needed here. Money fields ending `_inr_paise` (and
`inr_paise_per_kg`) are grounded in **both** their raw-paise form and a
rupee-crore conversion, since the rendered prompt (and any narrative
copying it) naturally states the crore figure, not raw paise.

**`llm_datapoints` (2026-09-02, Step 4 hardening, Concern 2) is
deliberately excluded from `render_facts_for_prompt` below** — the model
never sees a cited-but-not-independently-verified figure, so it can never
narrate one, full stop. This is the same "structured-only for now"
deferral this session's Step 3 hardening pass already applied to its own
new derived metrics: safely teaching a model to narrate a number while
visibly distinguishing "cited, not verified" from every other number in
this document is a real, separate design problem (how would a reader even
tell the two apart in prose without the narrative itself pointing it
out?) that deserves its own dedicated review, not a bullet point folded
into this pass. `flatten_facts_numbers` below still technically walks
`llm_datapoints`' numeric leaves (it's fully generic over `Facts`' shape),
which only makes `check_narrative_grounded` *more* permissive for those
specific numbers — harmless in practice, since a model that never sees a
number in its prompt has no way to guess it by coincidence."""

from __future__ import annotations

import contextlib
import re
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.budget import BudgetTracker
from app.guardrails import extract_numbers
from app.models import ModelClient
from app.report.facts import Facts

PROMPT_VERSION = "trade_narrative-v1"
_PROMPT_PATH = Path(__file__).resolve().parents[2] / "prompts" / "trade_narrative.md"

_PAISE_PER_RUPEE = Decimal(100)
_RUPEES_PER_CRORE = Decimal(10_000_000)
_PAISE_LEAF_SUFFIXES = ("_inr_paise", "inr_paise_per_kg")

_REGULATORY_WARNING_CLAUSE = (
    "\n\nOne additional hard rule for this document specifically: "
    "regulatory_note_missing_warning is true, meaning trade in this product is highly "
    "concentrated in one partner country and no verified regulatory explanation is on file. "
    "Describe this concentration neutrally (state the fact and the HHI/share given) and do "
    'not offer a commercial-preference explanation (e.g. "exporters prefer this partner", '
    '"this partner offers better prices") — you have no data to support such a claim, and a '
    "concentration this extreme is at least as often explained by an unrecorded regulatory "
    "restriction as by ordinary commercial preference."
)


class NarrativeOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    narrative: str = Field(min_length=1, max_length=2000)


@dataclass(frozen=True)
class NarrativeResult:
    narrative: str
    source: Literal["model", "model_retry", "template_fallback"]


def _load_system_prompt(*, regulatory_note_missing_warning: bool) -> str:
    text = _PROMPT_PATH.read_text(encoding="utf-8")
    if text.startswith("<!--"):
        end = text.find("-->")
        if end != -1:
            text = text[end + 3 :]
    text = text.strip()
    if regulatory_note_missing_warning:
        text += _REGULATORY_WARNING_CLAUSE
    return text


def _paise_to_crore(paise: int) -> Decimal:
    return Decimal(paise) / _PAISE_PER_RUPEE / _RUPEES_PER_CRORE


def _is_paise_leaf(key: str) -> bool:
    return any(key.endswith(suffix) for suffix in _PAISE_LEAF_SUFFIXES)


def flatten_facts_numbers(facts: Facts) -> set[float]:
    """Every numeric leaf of `facts`, rounded to 2 decimal places for
    tolerant comparison against prose-extracted numbers (matching
    `app.guardrails`' established rounding-tolerance precedent). A
    `_inr_paise`/`inr_paise_per_kg` leaf contributes both its raw-paise
    value and its rupee-crore conversion — either representation counts as
    grounded.

    `hs6` is added explicitly even though it's a `str` field, not a
    numeric one — the exact same real gap `app.guardrails._flatten_value_numbers`
    already documents and fixes for `hs_code`: a narrative legitimately
    says "HS6 120791", and `extract_numbers` has no way to tell that
    numeral apart from a fabricated trade figure. Without this, even this
    module's own deterministic `render_template_fallback` — which states
    the code because it's the subject of the whole document — would fail
    its own grounding check (caught by this module's own test suite, not
    inspection)."""
    grounded: set[float] = set()
    with contextlib.suppress(ValueError):
        grounded.add(float(facts.hs6))

    def _walk(key: str | None, value: object) -> None:
        if isinstance(value, bool):
            return  # bool is an int subtype in Python - never a real number here
        if isinstance(value, (int, float, Decimal)):
            grounded.add(round(float(value), 2))
            if key is not None and _is_paise_leaf(key) and isinstance(value, int):
                grounded.add(round(float(_paise_to_crore(value)), 2))
        elif isinstance(value, dict):
            for k, v in value.items():
                _walk(k, v)
        elif isinstance(value, list):
            for item in value:
                _walk(key, item)

    _walk(None, facts.model_dump(mode="python"))
    return grounded


_HS_LEVEL_LABEL_PATTERN = re.compile(r"\bHS(6|8)\b")


def _strip_hs_level_labels(text: str) -> str:
    """`extract_numbers`'s regex has no way to tell "the 6 in 'HS6'" (a
    classification-level label, not a digit adjacent to any other digit so
    its own hyphen/comma guards don't help) apart from a real, fabricated
    single-digit trade figure — caught by this module's own test suite: a
    fully grounded narrative (including this module's own deterministic
    `render_template_fallback`) that plainly writes "HS6 120791" would
    otherwise be rejected over the literal "6". "HS6"/"HS8" are removed
    down to "HS" before extraction; the real code number right after it
    (e.g. "120791") is untouched and still checked normally."""
    return _HS_LEVEL_LABEL_PATTERN.sub("HS", text)


def check_narrative_grounded(narrative: str, facts: Facts) -> bool:
    grounded = flatten_facts_numbers(facts)
    numbers = extract_numbers(_strip_hs_level_labels(narrative))
    return all(round(n, 2) in grounded for n in numbers)


def find_ungrounded_numbers(narrative: str, facts: Facts) -> list[float]:
    grounded = flatten_facts_numbers(facts)
    numbers = extract_numbers(_strip_hs_level_labels(narrative))
    return [n for n in numbers if round(n, 2) not in grounded]


def _render_year_line(year_entry: object) -> str:
    d = year_entry if isinstance(year_entry, dict) else {}
    total_paise = d.get("total_inr_paise")
    total_str = (
        f"₹{_paise_to_crore(total_paise):,.2f} crore" if total_paise is not None else "no total"
    )
    partners = d.get("partners") or []
    partner_str = "; ".join(
        f"#{p['rank']} {p['country']} ₹{_paise_to_crore(p['value_inr_paise']):,.2f} crore "
        f"({p['status']})"
        for p in partners
    )
    return f"  {d.get('year')} ({d.get('flow')}): total {total_str}, status {d.get('status')}" + (
        f" | {partner_str}" if partner_str else ""
    )


def render_facts_for_prompt(facts: Facts) -> str:
    """Compact plain-text rendering of `facts` for the model prompt — the
    only shape of trade data a model ever sees for this flow, matching
    `summarize.py`'s `render_table` precedent. Money is rendered in rupee
    crore (this pipeline's own native reporting unit, verified live
    against DGCIS's real report headers earlier this session), not raw
    paise."""
    dump = facts.model_dump(mode="python")
    lines = [
        f"HS code {facts.hs6} ({facts.product_label}), {facts.window.start_year}-"
        f"{facts.window.end_year} ({facts.window.years} years), top {facts.top_n} partners.",
        "",
        "Annual series:",
    ]
    lines.extend(_render_year_line(y) for y in dump["annual_series"])

    lines.append("")
    lines.append("Unit value trend (₹/kg) and FX decomposition:")
    for uv in dump["unit_value_trend"]:
        if uv["inr_paise_per_kg"] is None:
            lines.append(f"  {uv['year']}: not computable (coverage gate not passed)")
        else:
            lines.append(
                f"  {uv['year']}: ₹{uv['inr_paise_per_kg']:,.2f}/kg"
                f" | delta qty {uv['delta_qty_pct']}%, price {uv['delta_price_pct']}%,"
                f" fx {uv['delta_fx_pct']}%"
            )

    lines.append("")
    lines.append("Partner concentration (HHI, 0-1 scale):")
    for h in dump["hhi_by_year"]:
        lines.append(f"  {h['year']}: {'not computable' if h['hhi'] is None else h['hhi']}")

    lines.append("")
    lines.append("Cross-source mismatch checks:")
    if dump["mismatch_checks"]:
        for m in dump["mismatch_checks"]:
            lines.append(
                f"  {m['year']} {m['check']} vs {m['partner']}: {m['gap_pct']}% ({m['severity']})"
            )
    else:
        lines.append("  none computed for this window")

    lc = dump["landed_cost"]
    lines.append("")
    if lc is None:
        lines.append("Landed cost: not computable (no gate-passed unit-value year in window).")
    else:
        lines.append(
            f"Landed cost (as of {dump['landed_cost_as_of_period']}): "
            f"is_complete={lc['is_complete']}, "
            f"partial ₹{lc['partial_landed_cost_inr_paise_per_kg'] / 100:,.2f}/kg, "
            f"excluded components: {', '.join(lc['excluded_components']) or 'none'}."
        )
        for component, evidence in lc["components"].items():
            status = evidence["verification_status"]
            value = f"{evidence['value_pct']}%" if evidence["value_pct"] is not None else "n/a"
            lines.append(f"  {component}: {status} ({value})")

    lines.append("")
    if dump["regulatory_note"]:
        lines.append(f"Regulatory note: {dump['regulatory_note']}")
    else:
        lines.append(
            "Regulatory note: none on file."
            + (
                " (concentration warning: regulatory_note_missing_warning is true)"
                if dump["regulatory_note_missing_warning"]
                else ""
            )
        )

    cov = dump["coverage"]
    lines.append("")
    if cov is None:
        lines.append("Coverage: not assessed for this window.")
    else:
        lines.append(
            f"Coverage: {cov['present_cells']}/{cov['expected_cells']} cells present, "
            f"degraded={cov['degraded']}."
        )

    lines.append("")
    lines.append(dump["hs8_split_note"])

    return "\n".join(lines)


def render_template_fallback(facts: Facts) -> str:
    """Fully deterministic, code-generated narrative built directly from
    `facts` — no LLM call, grounded by construction. Used when both model
    attempts fail the grounding check, so a caller always gets a real
    narrative back rather than a bare error (§14's own "reject ->
    regenerate once -> template fallback" policy)."""
    sentences = [
        f"{facts.product_label} (HS code {facts.hs6}), {facts.flow}, "
        f"{facts.window.start_year}-{facts.window.end_year}.",
    ]
    for year_entry in facts.annual_series:
        if year_entry.total_inr_paise is None:
            sentences.append(f"{year_entry.year}: no data ({year_entry.status}).")
        else:
            crore = _paise_to_crore(year_entry.total_inr_paise)
            top_partner = year_entry.partners[0] if year_entry.partners else None
            partner_clause = ""
            if top_partner is not None:
                top_crore = _paise_to_crore(top_partner.value_inr_paise)
                partner_clause = f", led by {top_partner.country} at ₹{top_crore:,.2f} crore"
            sentences.append(
                f"{year_entry.year}: total ₹{crore:,.2f} crore "
                f"({year_entry.status}){partner_clause}."
            )
    if facts.landed_cost is not None:
        if facts.landed_cost.is_complete:
            sentences.append(
                f"Landed cost as of {facts.landed_cost_as_of_period}: "
                f"₹{(facts.landed_cost.landed_cost_inr_paise_per_kg or 0) / 100:,.2f}/kg."
            )
        else:
            sentences.append(
                f"Landed cost as of {facts.landed_cost_as_of_period} is incomplete "
                f"(unverified: {', '.join(facts.landed_cost.excluded_components)})."
            )
    if facts.regulatory_note:
        sentences.append(f"Regulatory note: {facts.regulatory_note}")
    elif facts.regulatory_note_missing_warning:
        sentences.append(
            "Trade is highly concentrated in one partner and no regulatory note is on file."
        )
    return " ".join(sentences)


async def generate_narrative(
    facts: Facts,
    *,
    model_client: ModelClient,
    budget_tracker: BudgetTracker,
    thread_id: str,
    tenant_id: str,
) -> NarrativeResult:
    """Budget is checked immediately before each real model call (matches
    `app.search.service.search_products`'s identical "check right before
    the spend it funds" sequencing for its own two-call retry path) — the
    retry is a second real spend, not a free do-over. `BudgetExceededError`
    is never caught here and propagates to the caller (matching
    `search_products`'s own precedent: budget exhaustion is a distinct,
    real failure the user should be told about via `BUDGET_EXCEEDED`, not
    silently masked behind a degraded-quality template narrative — the
    template fallback exists only for "the model tried and failed
    grounding," a different real failure mode)."""
    system_prompt = _load_system_prompt(
        regulatory_note_missing_warning=facts.regulatory_note_missing_warning
    )
    user_content = render_facts_for_prompt(facts)

    sources: tuple[Literal["model", "model_retry"], ...] = ("model", "model_retry")
    for source in sources:
        await budget_tracker.check_and_increment(thread_id=thread_id, tenant_id=tenant_id)
        result = await model_client.generate_structured(
            system_prompt=system_prompt, user_content=user_content, schema=NarrativeOutput
        )
        if check_narrative_grounded(result.narrative, facts):
            return NarrativeResult(narrative=result.narrative, source=source)

    return NarrativeResult(narrative=render_template_fallback(facts), source="template_fallback")
