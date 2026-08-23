"""Curator-facing CLI: upserts one HS6's free-text regulatory note
(`ref_regulatory_notes`) — import policy, notifications, compliance
requirements, chapter notes. Mirrors `scripts/record_duty_rate.py`'s
pattern: a human looks up the current official position and runs this
script with the real citation embedded in the note text itself (this
table has no separate source/citation columns of its own — the note is
expected to state its own evidence, or explicitly say what could not be
confirmed).

Evidence-first rule (same as duty rates): never fabricate a notification
number, compliance requirement, or policy status. If a claim can't be
confirmed from an authoritative source, the note should say so explicitly
rather than omit the caveat.

Usage:

    uv run python scripts/record_regulatory_note.py \\
      --hs6 120791 --updated-by "Claude Sonnet 5 (curator session 2026-08-24)" \\
      --note "$(cat note.txt)"
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from sqlalchemy import func
from sqlalchemy.dialects.postgresql import insert

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.warehouse.db import get_engine
from app.warehouse.schema import ref_regulatory_notes


def _build_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--hs6", required=True)
    parser.add_argument("--note", required=True)
    parser.add_argument("--updated-by", required=True)
    return parser.parse_args(argv)


async def record_regulatory_note(args: argparse.Namespace) -> None:
    engine = get_engine()
    async with engine.begin() as conn:
        stmt = insert(ref_regulatory_notes).values(
            hs6=args.hs6, note=args.note, updated_by=args.updated_by
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=["hs6"],
            set_={
                "note": stmt.excluded.note,
                "updated_by": stmt.excluded.updated_by,
                "updated_at": func.now(),
            },
        )
        await conn.execute(stmt)
    print(f"Recorded regulatory note for HS6 {args.hs6}")


async def main() -> None:
    args = _build_args()
    await record_regulatory_note(args)


if __name__ == "__main__":
    asyncio.run(main())
