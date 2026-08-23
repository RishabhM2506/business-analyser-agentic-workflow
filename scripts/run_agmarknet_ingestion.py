"""Real, curator-facing CLI: runs `app.pipeline.agmarknet.fetch_all_records`
for one commodity/state/district filter set and upserts the results into
`raw_agmarknet_prices`. Mirrors `scripts/run_dgcis_country_batch.py`'s
shape: any real fetch failure (the retry schedule exhausted) is written
to `dead_letter_ingestion` (D3) rather than silently dropped, and records
are upserted in batches as they stream in rather than held entirely in
memory (this dataset is real and large - 81M+ total rows).

Usage (from the repo root, with `DATABASE_URL` and `AGMARKNET_API_KEY`
set, e.g. via `.env`):

    # Always filter by at least commodity or state/district first - an
    # unfiltered run would attempt to page through 81M+ real rows.
    uv run python scripts/run_agmarknet_ingestion.py \\
      --commodity "Poppy seeds" --limit 500

    uv run python scripts/run_agmarknet_ingestion.py \\
      --state "Madhya Pradesh" --district Neemuch --limit 500
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx
from sqlalchemy import insert

from app.pipeline.agmarknet import (
    BASE_URL,
    AgmarknetError,
    AgmarknetRecord,
    fetch_all_records,
    upsert_agmarknet_records,
)
from app.settings import get_settings
from app.warehouse.db import get_engine
from app.warehouse.schema import dead_letter_ingestion

_UPSERT_BATCH_SIZE = 500


def _build_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--commodity", default=None, help="exact Agmarknet Commodity value")
    parser.add_argument("--state", default=None, help="exact Agmarknet State value")
    parser.add_argument("--district", default=None, help="exact Agmarknet District value")
    parser.add_argument(
        "--limit", type=int, default=None, help="stop after this many real records fetched"
    )
    parser.add_argument("--page-size", type=int, default=100)
    return parser.parse_args(argv)


async def main() -> None:
    args = _build_args()
    if not (args.commodity or args.state or args.district):
        raise SystemExit(
            "at least one of --commodity/--state/--district is required "
            "(an unfiltered run would page through 81M+ real rows)"
        )

    settings = get_settings()
    engine = get_engine()
    job_run_id = uuid.uuid4()
    started = datetime.now(UTC)

    fetched = 0
    rows_written = 0
    batch: list[AgmarknetRecord] = []

    async def _flush() -> None:
        nonlocal rows_written, batch
        if not batch:
            return
        rows_written += await upsert_agmarknet_records(engine, batch)
        batch = []

    async with httpx.AsyncClient(base_url=BASE_URL, timeout=30.0) as client:
        try:
            async for record in fetch_all_records(
                client,
                api_key=settings.agmarknet_api_key,
                commodity=args.commodity,
                state=args.state,
                district=args.district,
                page_size=args.page_size,
            ):
                batch.append(record)
                fetched += 1
                if len(batch) >= _UPSERT_BATCH_SIZE:
                    await _flush()
                print(f"fetched={fetched} {record.price_date} {record.commodity} {record.market}")
                if args.limit is not None and fetched >= args.limit:
                    break
            await _flush()
        except AgmarknetError as exc:
            await _flush()
            async with engine.begin() as conn:
                await conn.execute(
                    insert(dead_letter_ingestion),
                    [
                        {
                            "source": "agmarknet",
                            "job_run_id": job_run_id,
                            "attempted_at": datetime.now(UTC),
                            "request_desc": (
                                f"commodity={args.commodity} state={args.state} "
                                f"district={args.district} fetched_before_failure={fetched}"
                            ),
                            "error_message": str(exc),
                            "attempt_count": 1,
                            "resolved": False,
                        }
                    ],
                )
            print(f"FAIL: {exc}", flush=True)

    elapsed = (datetime.now(UTC) - started).total_seconds()
    print(
        f"\nDONE. fetched={fetched} rows_written={rows_written} elapsed_s={elapsed:.1f} "
        f"job_run_id={job_run_id}",
        flush=True,
    )
    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
