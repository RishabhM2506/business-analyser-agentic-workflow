"""Curator-facing CLI: searches for a real, citable value for a `Facts`
field the verified analytics/ref layer genuinely has nothing for
(`mandi_price`/`msp`/`international_production` today —
`LLM_DATAPOINT_FIELDS`), writes it to `ref_llm_datapoints` if — and only
if — a real citation comes back with it (2026-09-02, Step 4 hardening,
Concern 2).

This is the *only* write path into `ref_llm_datapoints` — like
`record_duty_rate.py`, no automated ingestion job populates it. Unlike
that script, the search itself is the point (a human isn't looking the
value up directly), but the same "never fabricate" discipline applies:
`app.models.GeminiModelClient.generate_grounded`'s two-call design raises
`UngroundedSearchError` when no real citation comes back, and this script
treats that as a normal, expected "nothing found this time" outcome — it
skips the field and moves on, never writes an uncited row.

**No approval gate** (2026-09-02, user-directed correction to an earlier,
more cautious draft of this feature): a written row is immediately live
and readable by `assemble_facts` — the citation columns are the safety
mechanism, not a human sign-off step, matching the `--dry-run` flag below
being the *pre*-write check a curator can use if they want one, not a
required one.

Refuses to run at all under `LLM_PROVIDER=mock` (`Settings.llm_provider`)
— `MockLLM.generate_grounded`'s deterministic fake citation exists so CI
never makes a real network call, not so a curator can accidentally write
"Mock citation" rows into a real warehouse.

Usage examples (from the repo root, with `DATABASE_URL` and a real
`GEMINI_API_KEY`/`GEMINI_API_KEYS_EXTRA` pointed at the real things):

    # Search every currently-missing field for this product.
    uv run python scripts/run_llm_datapoint_search.py --hs6 120791

    # Just one field, and preview what would be searched without writing.
    uv run python scripts/run_llm_datapoint_search.py \\
      --hs6 120791 --fields mandi_price --dry-run

    # A small first run against a handful of products, matching
    # run_dgcis_country_batch.py's own "small validation batch first"
    # guidance before a larger run.
    uv run python scripts/run_llm_datapoint_search.py --hs6 120791 --limit 1
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict
from sqlalchemy import insert
from sqlalchemy.ext.asyncio import AsyncEngine

# See scripts/record_duty_rate.py's own identical comment for why this is
# needed at all.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.knowledge.provider import get_taxonomy_entry
from app.models import GroundedResult, ModelClient, UngroundedSearchError, get_model_for_role
from app.report.facts import (
    InternationalProductionFact,
    MandiPriceFact,
    MspFact,
    _fetch_international_production,
    _fetch_mandi_price,
    _fetch_msp,
)
from app.settings import get_settings
from app.warehouse.db import get_engine
from app.warehouse.schema import LLM_DATAPOINT_FIELDS, ref_llm_datapoints

FieldName = Literal["mandi_price", "msp", "international_production"]


class RunLlmDatapointSearchError(Exception):
    """Raised for any usage/data problem — never silently proceeds."""


class _MandiPriceSearchResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    modal_price_inr_paise_per_qtl: int
    market: str
    price_date: str


class _MspSearchResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    msp_inr_paise_per_qtl: int
    year_label: str


class _InternationalProductionSearchResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    india_production_tonnes: int
    world_production_tonnes: int
    year: int


_SEARCH_SCHEMAS: dict[FieldName, type[BaseModel]] = {
    "mandi_price": _MandiPriceSearchResult,
    "msp": _MspSearchResult,
    "international_production": _InternationalProductionSearchResult,
}

_SEARCH_PROMPTS: dict[FieldName, str] = {
    "mandi_price": (
        "Find the most recent modal (most common) wholesale price for {product} "
        "at a real Indian agricultural mandi (wholesale market), from a real, "
        "checkable source such as Agmarknet or a market bulletin. State the "
        "price in INR per quintal, the market name, and the date it was "
        "recorded."
    ),
    "msp": (
        "Find India's most recent official Minimum Support Price (MSP) for "
        "{product}, as announced by the Government of India (CACP/Ministry of "
        "Agriculture), from a real, checkable source. State the MSP in INR "
        "per quintal and the marketing year it applies to (e.g. '2025-26')."
    ),
    "international_production": (
        "Find India's production volume and the world's total production "
        "volume of {product} for the most recent year real data exists, from "
        "a real, checkable source such as FAOSTAT. State both figures in "
        "metric tonnes and the year they refer to."
    ),
}


async def _already_verified(engine: AsyncEngine, *, field: FieldName, product_label: str) -> bool:
    """True iff the verified analytics/ref layer already has a real value
    for this field — searching would be pointless spend against a field
    that isn't actually missing."""
    if field == "mandi_price":
        fact: MandiPriceFact | MspFact | InternationalProductionFact = await _fetch_mandi_price(
            engine, taxonomy_description=product_label
        )
    elif field == "msp":
        fact = await _fetch_msp(engine, taxonomy_description=product_label)
    else:
        fact = await _fetch_international_production(engine, taxonomy_description=product_label)
    return fact.status not in ("NOT_FOUND", "NOT_APPLICABLE")


def _build_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--hs6", required=True)
    parser.add_argument(
        "--fields",
        default=",".join(LLM_DATAPOINT_FIELDS),
        help=f"comma-separated subset of {LLM_DATAPOINT_FIELDS}",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="search at most this many fields this run (a small-batch-first safety valve)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="search and print, but never write to the database"
    )
    return parser.parse_args(argv)


def _parse_fields(raw: str) -> list[FieldName]:
    fields = [f.strip() for f in raw.split(",") if f.strip()]
    invalid = [f for f in fields if f not in LLM_DATAPOINT_FIELDS]
    if invalid:
        raise RunLlmDatapointSearchError(
            f"--fields contains unsupported field(s) {invalid!r}, must be a subset of "
            f"{LLM_DATAPOINT_FIELDS}"
        )
    return fields  # type: ignore[return-value]


async def run_llm_datapoint_search(args: argparse.Namespace) -> None:
    settings = get_settings()
    if settings.llm_provider == "mock":
        raise RunLlmDatapointSearchError(
            "LLM_PROVIDER=mock — refusing to run. MockLLM's citations are fixed, obviously-fake "
            "placeholders for CI, not real search results; running this script under mock mode "
            "would write fabricated-looking rows into the real warehouse."
        )

    fields = _parse_fields(args.fields)
    engine = get_engine()
    taxonomy_entry = get_taxonomy_entry(args.hs6)
    product_label = taxonomy_entry.description if taxonomy_entry is not None else args.hs6

    to_search: list[FieldName] = []
    for field in fields:
        if await _already_verified(engine, field=field, product_label=product_label):
            print(f"{args.hs6}/{field}: already has a real verified value — skipping search")
            continue
        to_search.append(field)
    if args.limit is not None:
        to_search = to_search[: args.limit]

    if not to_search:
        print(f"{args.hs6}: nothing to search")
        return

    model_client: ModelClient = get_model_for_role("utility", provider=settings.llm_provider)

    for field in to_search:
        prompt = _SEARCH_PROMPTS[field].format(product=product_label)
        print(f"{args.hs6}/{field}: searching — {prompt}")
        if args.dry_run:
            continue
        try:
            result: GroundedResult[BaseModel] = await model_client.generate_grounded(
                system_prompt=(
                    "You are researching real, current Indian agricultural trade data. "
                    "Only state facts you found via search — never estimate or guess."
                ),
                user_content=prompt,
                schema=_SEARCH_SCHEMAS[field],
            )
        except UngroundedSearchError as exc:
            print(f"{args.hs6}/{field}: no real citation found — skipping ({exc})")
            continue

        citation = result.citations[0]
        extra_sources = (
            "; also cited: " + ", ".join(c.source_url for c in result.citations[1:])
            if len(result.citations) > 1
            else None
        )
        async with engine.begin() as conn:
            await conn.execute(
                insert(ref_llm_datapoints).values(
                    hs6=args.hs6,
                    field_name=field,
                    effective_period=datetime.now(UTC).strftime("%Y-%m"),
                    value_json=result.value.model_dump(),
                    source_authority=citation.title or citation.source_url,
                    source_reference=citation.source_url,
                    source_url=citation.source_url,
                    verified_date=datetime.now(UTC).date(),
                    notes=extra_sources,
                )
            )
        print(f"{args.hs6}/{field}: wrote a real, cited datapoint from {citation.source_url}")


async def main() -> None:
    args = _build_args()
    await run_llm_datapoint_search(args)


if __name__ == "__main__":
    asyncio.run(main())
