"""Real, curator-facing CLI: runs `app.pipeline.dgcis.fetch_year_monthly`
for one (hs8, flow, year), upserts into `raw_dgcis_monthly`, then computes
and upserts `analytics_monthly_current_year` for the same (hs6, flow,
year) via `app.report.monthly_current_year`. Mirrors
`scripts/run_dgcis_country_batch.py`'s shape for the monthly path.

Any real fetch failure is written to `dead_letter_ingestion` (D3) rather
than silently dropped — the batch itself never aborts on one month's
failure (`fetch_year_monthly`'s own contract). A month with no matching
row in DGCIS's response (as opposed to a real `(A)`-marked "not yet
published" row) is neither a failure nor a written row — it surfaces as
`NOT_YET_PUBLISHED` at the analytics layer regardless, per
`app.report.monthly_current_year`'s own documented "can't tell the two
apart at this layer" limitation.

Usage (from the repo root, with `DATABASE_URL` pointed at a real
Postgres):

    uv run python scripts/run_dgcis_monthly.py \\
      --hs8 12079100 --hs6 120791 --flow import --year 2026
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
    MONTHLY_EXPORT_PATH,
    MONTHLY_IMPORT_PATH,
    DgcisClient,
    DgcisMonthlyFetchFailure,
    fetch_year_monthly,
    upsert_monthly_records,
)
from app.report.monthly_current_year import (
    compute_monthly_current_year,
    upsert_monthly_current_year,
)
from app.warehouse.db import get_engine
from app.warehouse.schema import dead_letter_ingestion


def _build_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--hs8", required=True, help="8-digit ITC-HS code, e.g. 12079100")
    parser.add_argument("--hs6", required=True, help="6-digit HS6 code, e.g. 120791")
    parser.add_argument("--flow", required=True, choices=("import", "export"))
    parser.add_argument("--year", required=True, type=int)
    parser.add_argument("--delay-seconds", type=float, default=1.0)
    return parser.parse_args(argv)


async def main() -> None:
    args = _build_args()
    path = MONTHLY_IMPORT_PATH if args.flow == "import" else MONTHLY_EXPORT_PATH

    engine = get_engine()
    client = DgcisClient()
    job_run_id = uuid.uuid4()

    succeeded_months: list[int] = []
    failures: list[DgcisMonthlyFetchFailure] = []
    rows_written = 0
    started = datetime.now(UTC)

    try:
        async for item in fetch_year_monthly(
            client, path=path, hs8=args.hs8, year=args.year, delay_seconds=args.delay_seconds
        ):
            if isinstance(item, DgcisMonthlyFetchFailure):
                failures.append(item)
                print(f"FAIL month={item.month} {item.year}: {item.error}", flush=True)
                continue
            n = await upsert_monthly_records(engine, [item])
            succeeded_months.append(item.calendar_month.month)
            rows_written += n
            print(
                f"OK month={item.calendar_month.month} marker={item.marker!r} "
                f"value_inr_paise={item.value_inr_paise} quantity_kg={item.quantity_kg}",
                flush=True,
            )
    finally:
        await client.aclose()

    if failures:
        async with engine.begin() as conn:
            await conn.execute(
                insert(dead_letter_ingestion),
                [
                    {
                        "source": "dgcis_monthly",
                        "job_run_id": job_run_id,
                        "attempted_at": datetime.now(UTC),
                        "request_desc": f"{path} hs8={args.hs8} month={f.month} year={f.year}",
                        "error_message": f.error,
                        "attempt_count": 1,
                        "resolved": False,
                    }
                    for f in failures
                ],
            )

    analytics_rows = await compute_monthly_current_year(
        engine, hs6=args.hs6, flow=args.flow, year=args.year
    )
    analytics_written = await upsert_monthly_current_year(
        engine, analytics_rows, data_as_of=datetime.now(UTC)
    )
    for row in analytics_rows:
        print(
            f"ANALYTICS {row.month} status={row.status} value_inr_paise={row.value_inr_paise} "
            f"is_provisional={row.is_provisional} mom={row.mom_change_pct} "
            f"yoy={row.yoy_same_month_pct}",
            flush=True,
        )

    elapsed = (datetime.now(UTC) - started).total_seconds()
    print(
        f"\nDONE. months_with_data={len(succeeded_months)}/12 failed={len(failures)} "
        f"raw_rows_written={rows_written} analytics_rows_written={analytics_written} "
        f"elapsed_s={elapsed:.1f} job_run_id={job_run_id}",
        flush=True,
    )
    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
