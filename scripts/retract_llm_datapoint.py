"""Curator-facing CLI: retracts one `ref_llm_datapoints` row after the
fact (2026-09-02, Step 4 hardening, Concern 2) — the correction path this
feature has instead of a pre-publication approval gate (see
`run_llm_datapoint_search.py`'s own docstring for why there's no gate).
A retracted row is excluded from `assemble_facts`'s `llm_datapoints`
immediately; it is never deleted, so the record of what was found and
later found to be wrong stays auditable.

Usage (from the repo root, with `DATABASE_URL` pointed at a real Postgres):

    uv run python scripts/retract_llm_datapoint.py --list --hs6 120791
    uv run python scripts/retract_llm_datapoint.py \\
      --id 42 --reason "Source page was actually about a different commodity"
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from sqlalchemy import select, update

# See scripts/record_duty_rate.py's own identical comment for why this is
# needed at all.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.warehouse.db import get_engine
from app.warehouse.schema import ref_llm_datapoints


class RetractLlmDatapointError(Exception):
    """Raised for any usage/data problem — never silently proceeds."""


def _build_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--list", action="store_true", help="list ACTIVE rows for --hs6, then exit")
    parser.add_argument("--hs6", default=None, help="required with --list")
    parser.add_argument("--id", type=int, default=None, help="the row id to retract")
    parser.add_argument("--reason", default=None, help="required with --id")
    return parser.parse_args(argv)


async def _list_active(hs6: str) -> None:
    engine = get_engine()
    async with engine.connect() as conn:
        rows = (
            (
                await conn.execute(
                    select(ref_llm_datapoints).where(
                        ref_llm_datapoints.c.hs6 == hs6,
                        ref_llm_datapoints.c.status == "ACTIVE",
                    )
                )
            )
            .mappings()
            .all()
        )
    if not rows:
        print(f"{hs6}: no ACTIVE ref_llm_datapoints rows")
        return
    for r in rows:
        print(
            f"id={r['id']} field={r['field_name']} effective_period={r['effective_period']} "
            f"value={r['value_json']!r} source={r['source_reference']} "
            f"({r['source_url'] or 'no URL'}) verified_date={r['verified_date']}"
        )


async def _retract(row_id: int, reason: str) -> None:
    engine = get_engine()
    async with engine.begin() as conn:
        result = await conn.execute(
            update(ref_llm_datapoints)
            .where(ref_llm_datapoints.c.id == row_id, ref_llm_datapoints.c.status == "ACTIVE")
            .values(status="RETRACTED", retracted_reason=reason)
        )
    if result.rowcount == 0:
        raise RetractLlmDatapointError(
            f"id={row_id}: no matching ACTIVE row (already retracted, or id doesn't exist)"
        )
    print(f"id={row_id}: retracted — {reason}")


async def retract_llm_datapoint(args: argparse.Namespace) -> None:
    if args.list:
        if not args.hs6:
            raise RetractLlmDatapointError("--list requires --hs6")
        await _list_active(args.hs6)
        return
    if args.id is None or not args.reason:
        raise RetractLlmDatapointError("retracting a row requires both --id and --reason")
    await _retract(args.id, args.reason)


async def main() -> None:
    args = _build_args()
    await retract_llm_datapoint(args)


if __name__ == "__main__":
    asyncio.run(main())
