"""Real, curator-facing CLI: runs `app.pipeline.dgcis.fetch_all_countries_annual`
across the tracked country list for one (hs8, flow, year) and upserts the
results into `raw_dgcis_annual`. This is the script that produced the
first real, full ~250-country DGCIS run for HS6 120791 (`docs/BUILD-LOG.md`).

Any real fetch failure is written to `dead_letter_ingestion` (D3) rather
than silently dropped — the batch itself never aborts on one country's
failure (`fetch_all_countries_annual`'s own contract).

Usage (from the repo root, with `DATABASE_URL` pointed at a real
Postgres):

    # Small validation batch first - always run this before a full list,
    # to observe real rate-limit/throttling behavior on however many
    # countries you're about to hit (docs/pipeline/dgcis.py's own
    # "conservative and unverified" flag on the default delay).
    uv run python scripts/run_dgcis_country_batch.py \\
      --hs8 12079100 --flow import --year 2024 --limit 15

    # Full tracked list.
    uv run python scripts/run_dgcis_country_batch.py \\
      --hs8 12079100 --flow import --year 2024
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import insert

from app.pipeline.dgcis import (
    ANNUAL_EXPORT_PATH,
    ANNUAL_IMPORT_PATH,
    DgcisClient,
    DgcisFetchFailure,
    fetch_all_countries_annual,
    get_dgcis_countries,
    upsert_annual_records,
)
from app.warehouse.db import get_engine
from app.warehouse.schema import dead_letter_ingestion


def _build_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--hs8", required=True, help="8-digit ITC-HS code, e.g. 12079100")
    parser.add_argument("--flow", required=True, choices=("import", "export"))
    parser.add_argument(
        "--year", required=True, help='report "ending year", e.g. "2024" for FY2020-21..2024-25'
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="process only the first N tracked countries (for a validation batch)",
    )
    parser.add_argument("--delay-seconds", type=float, default=1.0)
    return parser.parse_args(argv)


async def main() -> None:
    args = _build_args()
    path = ANNUAL_IMPORT_PATH if args.flow == "import" else ANNUAL_EXPORT_PATH
    countries = get_dgcis_countries()
    if args.limit is not None:
        countries = countries[: args.limit]

    engine = get_engine()
    client = DgcisClient()
    job_run_id = uuid.uuid4()

    succeeded = 0
    rows_written = 0
    failures: list[DgcisFetchFailure] = []
    started = datetime.now(UTC)

    try:
        async for item in fetch_all_countries_annual(
            client,
            path=path,
            hs8=args.hs8,
            year=args.year,
            countries=countries,
            delay_seconds=args.delay_seconds,
        ):
            if isinstance(item, DgcisFetchFailure):
                failures.append(item)
                print(f"FAIL {item.country.code} {item.country.name}: {item.error}", flush=True)
                continue
            n = await upsert_annual_records(engine, [item], flow=args.flow)
            succeeded += 1
            rows_written += n
            progress = succeeded + len(failures)
            print(f"OK {item.country} rows={n} (progress {progress}/{len(countries)})", flush=True)
    finally:
        await client.aclose()

    if failures:
        async with engine.begin() as conn:
            await conn.execute(
                insert(dead_letter_ingestion),
                [
                    {
                        "source": "dgcis",
                        "job_run_id": job_run_id,
                        "attempted_at": datetime.now(UTC),
                        "request_desc": (
                            f"{path} hs8={args.hs8} country={f.country.code} "
                            f"({f.country.name}) year={args.year}"
                        ),
                        "error_message": f.error,
                        "attempt_count": 1,
                        "resolved": False,
                    }
                    for f in failures
                ],
            )

    elapsed = (datetime.now(UTC) - started).total_seconds()
    print(
        f"\nDONE. countries_attempted={len(countries)} succeeded={succeeded} "
        f"failed={len(failures)} rows_written={rows_written} elapsed_s={elapsed:.1f} "
        f"job_run_id={job_run_id}",
        flush=True,
    )
    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
