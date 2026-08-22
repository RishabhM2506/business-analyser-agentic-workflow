"""Curator-facing CLI: records one verified/not-verified/conflicting duty
component for one HS8 line (`docs/PLAN.md` §4a). This is the *only*
write path into `ref_duty_components`/`ref_duty_component_conflicts` — no
ingestion job populates these tables, since ICEGATE's Trade Guide on
Imports and the CBIC Tax Information Portal (the only sources this system
trusts) expose no public API. A human looks up the current rate there and
runs this script with the real citation.

Usage examples (from the repo root, with `DATABASE_URL` pointed at a real
Postgres):

    # A verified rate, real citation required.
    uv run python scripts/record_duty_rate.py \\
      --hs8 12079100 --component BCD --status VERIFIED --value-pct 20.0 \\
      --source-authority "ICEGATE Trade Guide on Imports" \\
      --source-reference "Notification No. 50/2017-Customs, as amended" \\
      --source-url "https://www.icegate.gov.in/services/customs-duty-calculator" \\
      --effective-from 2024-02-01 --verified-date 2026-08-23

    # No authoritative source found - never fabricate a value instead.
    uv run python scripts/record_duty_rate.py \\
      --hs8 12079100 --component SWS --status NOT_VERIFIED \\
      --effective-from 2026-08-23 --verified-date 2026-08-23

    # Two official sources disagree - both recorded, neither auto-picked.
    uv run python scripts/record_duty_rate.py \\
      --hs8 12079100 --component IGST --status CONFLICTING \\
      --conflict 5.0 "CBIC Tax Information Portal" "Notification A" \\
      --conflict 12.0 "CBIC Tax Information Portal" "Notification B" \\
      --effective-from 2026-08-23 --verified-date 2026-08-23

Entering a new row for a component that already has a current one
(`effective_to IS NULL`) atomically closes out the old row
(`effective_to` set to the new row's `effective_from`) in the same
transaction — and if the old row was itself `VERIFIED`, flips it to
`EXPIRED` (a real number that's now superseded, preserved for historical
analysis per the user's own rule). A superseded `NOT_VERIFIED`/
`CONFLICTING` row is only closed out, not relabeled `EXPIRED` — it never
had a real verified number to begin with, so "expired" would misdescribe
it; it stays historically accurate as whatever it was.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import date
from decimal import Decimal, InvalidOperation
from pathlib import Path

from sqlalchemy import insert, select, update
from sqlalchemy.ext.asyncio import AsyncConnection

# This script lives outside `app/`'s package tree (matches
# scripts/embed_taxonomy.py's own established pattern) - `uv run python
# scripts/foo.py` doesn't put the repo root on `sys.path` for absolute
# `app.*` imports, so it's added explicitly here, once, before those
# imports. Live-reproduced: without this, `uv run python
# scripts/record_duty_rate.py` fails with `ModuleNotFoundError: No module
# named 'app'`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.pipeline.duty_source import DUTY_COMPONENT_VALUES
from app.warehouse.db import get_engine
from app.warehouse.schema import ref_duty_component_conflicts, ref_duty_components


class RecordDutyRateError(Exception):
    """Raised for any usage/data problem — never silently proceeds with a
    guessed or partially-valid record."""


def _parse_date(value: str) -> date:
    return date.fromisoformat(value)


def _parse_conflict(raw: list[str]) -> tuple[Decimal, str, str, str | None]:
    if len(raw) not in (3, 4):
        raise RecordDutyRateError(
            f"--conflict expects 3 or 4 values (value_pct, source_authority, source_reference"
            f"[, source_url]), got {len(raw)}: {raw!r}"
        )
    try:
        value_pct = Decimal(raw[0])
    except InvalidOperation as exc:
        raise RecordDutyRateError(f"--conflict value_pct is not a number: {raw[0]!r}") from exc
    source_url = raw[3] if len(raw) == 4 else None
    return value_pct, raw[1], raw[2], source_url


def _build_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--hs8", required=True)
    parser.add_argument("--component", required=True, choices=DUTY_COMPONENT_VALUES)
    parser.add_argument(
        "--status", required=True, choices=("VERIFIED", "NOT_VERIFIED", "CONFLICTING")
    )
    parser.add_argument("--value-pct", type=Decimal, default=None)
    parser.add_argument(
        "--conflict",
        nargs="+",
        action="append",
        default=[],
        metavar="VALUE_PCT SOURCE_AUTHORITY SOURCE_REFERENCE [SOURCE_URL]",
    )
    parser.add_argument("--source-authority", default=None)
    parser.add_argument("--source-reference", default=None)
    parser.add_argument("--source-url", default=None)
    parser.add_argument("--effective-from", required=True, type=_parse_date)
    parser.add_argument("--verified-date", required=True, type=_parse_date)
    parser.add_argument("--notes", default=None)
    return parser.parse_args(argv)


def _validate(args: argparse.Namespace) -> None:
    if args.status in ("VERIFIED",):
        if args.value_pct is None:
            raise RecordDutyRateError(f"--status {args.status} requires --value-pct")
        if not args.source_authority or not args.source_reference:
            raise RecordDutyRateError(
                f"--status {args.status} requires --source-authority and --source-reference "
                f"(never a value without a citation)"
            )
        if args.conflict:
            raise RecordDutyRateError(f"--status {args.status} must not have --conflict entries")
    elif args.status == "NOT_VERIFIED":
        if args.value_pct is not None or args.conflict:
            raise RecordDutyRateError(
                "--status NOT_VERIFIED must not have --value-pct or --conflict "
                "(there is no trustworthy number to record)"
            )
        # ref_duty_components.source_authority/source_reference are NOT
        # NULL (schema.py's own comment anticipates this exact case: "or
        # 'none found' for NOT_VERIFIED") — live-reproduced: omitting them
        # entirely violates the DB constraint, not just a style choice.
        if not args.source_authority:
            args.source_authority = "none found"
        if not args.source_reference:
            args.source_reference = "none found"
    elif args.status == "CONFLICTING":
        if args.value_pct is not None:
            raise RecordDutyRateError(
                "--status CONFLICTING must not have --value-pct — the whole point is that no "
                "single value is automatically chosen"
            )
        if len(args.conflict) < 2:
            raise RecordDutyRateError(
                "--status CONFLICTING requires at least 2 --conflict entries "
                "(a single candidate is not a conflict)"
            )
        if not args.source_authority:
            args.source_authority = "multiple (see conflicting candidates)"
        if not args.source_reference:
            args.source_reference = "see conflicting candidates"


async def _close_out_previous_row(
    conn: AsyncConnection, *, hs8: str, component: str, new_effective_from: date
) -> None:
    current_stmt = select(ref_duty_components).where(
        ref_duty_components.c.hs8 == hs8,
        ref_duty_components.c.component == component,
        ref_duty_components.c.effective_to.is_(None),
    )
    current = (await conn.execute(current_stmt)).mappings().one_or_none()
    if current is None:
        return
    if current["effective_from"] >= new_effective_from:
        raise RecordDutyRateError(
            f"{hs8}/{component}: an existing row already covers {current['effective_from']} "
            f"onward — the new row's --effective-from ({new_effective_from}) must be strictly "
            f"later, to keep effective_from a real, ordered timeline."
        )
    new_status = (
        "EXPIRED"
        if current["verification_status"] == "VERIFIED"
        else current["verification_status"]
    )
    await conn.execute(
        update(ref_duty_components)
        .where(
            ref_duty_components.c.hs8 == hs8,
            ref_duty_components.c.component == component,
            ref_duty_components.c.effective_from == current["effective_from"],
        )
        .values(effective_to=new_effective_from, verification_status=new_status)
    )


async def record_duty_rate(args: argparse.Namespace) -> None:
    _validate(args)
    engine = get_engine()
    async with engine.begin() as conn:
        await _close_out_previous_row(
            conn, hs8=args.hs8, component=args.component, new_effective_from=args.effective_from
        )
        await conn.execute(
            insert(ref_duty_components).values(
                hs8=args.hs8,
                component=args.component,
                effective_from=args.effective_from,
                effective_to=None,
                verification_status=args.status,
                value_pct=args.value_pct,
                source_authority=args.source_authority,
                source_reference=args.source_reference,
                source_url=args.source_url,
                verified_date=args.verified_date,
                notes=args.notes,
            )
        )
        if args.status == "CONFLICTING":
            candidates = [_parse_conflict(raw) for raw in args.conflict]
            await conn.execute(
                insert(ref_duty_component_conflicts),
                [
                    {
                        "hs8": args.hs8,
                        "component": args.component,
                        "effective_from": args.effective_from,
                        "candidate_value_pct": value_pct,
                        "source_authority": source_authority,
                        "source_reference": source_reference,
                        "source_url": source_url,
                    }
                    for value_pct, source_authority, source_reference, source_url in candidates
                ],
            )
    print(
        f"Recorded {args.component} for {args.hs8}: {args.status}, effective {args.effective_from}"
    )


async def main() -> None:
    args = _build_args()
    await record_duty_rate(args)


if __name__ == "__main__":
    asyncio.run(main())
