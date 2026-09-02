"""Real, curator-facing CLI: runs `app.pipeline.msp.fetch_all_records`
and upserts the results into `raw_msp_records`. Mirrors
`scripts/run_agmarknet_ingestion.py`'s shape. Unlike Agmarknet, this
resource is a small, static reference table (22 real rows as of
2026-08-24) - no `--limit`/filter flags are needed, the whole dataset is
fetched every run.

Usage (from the repo root, with `DATABASE_URL` and `AGMARKNET_API_KEY`
set, e.g. via `.env` - the same data.gov.in key covers every resource on
this platform, Agmarknet included):

    uv run python scripts/run_msp_ingestion.py
"""

from __future__ import annotations

import asyncio
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx

from app.pipeline.data_gov_in import BASE_URL, DataGovInError
from app.pipeline.msp import fetch_all_records, upsert_msp_records
from app.settings import get_settings
from app.warehouse.db import get_engine


async def main() -> None:
    settings = get_settings()
    engine = get_engine()
    started = datetime.now(UTC)

    async with httpx.AsyncClient(base_url=BASE_URL, timeout=30.0) as client:
        try:
            records = await fetch_all_records(client, api_key=settings.agmarknet_api_key)
        except DataGovInError as exc:
            print(f"FAIL: {exc}", flush=True)
            raise SystemExit(1) from exc

    written = await upsert_msp_records(engine, records)
    elapsed = (datetime.now(UTC) - started).total_seconds()
    print(
        f"DONE. fetched={len(records)} rows_written={written} elapsed_s={elapsed:.1f}",
        flush=True,
    )
    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
