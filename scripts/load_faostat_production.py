"""Real, curator-facing CLI: downloads FAOSTAT's real production bulk ZIP
(no auth, no rate limit - see `app.pipeline.faostat`'s module docstring
for why this is the bulk-download path rather than the REST API, which is
currently down) and loads real records for the given item name(s) into
`raw_faostat_records`.

Usage (from the repo root, with `DATABASE_URL` pointed at a real
Postgres):

    uv run python scripts/load_faostat_production.py --item "Poppy seed"

`--item` is a real FAOSTAT `Item` string (case-sensitive, exact match) -
verify it against the real `Production_Crops_Livestock_E_ItemCodes.csv`
member inside the downloaded ZIP if unsure; this script does not guess or
fuzzy-match item names.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx

from app.pipeline.faostat import PRODUCTION_ZIP_URL, load_production_zip, upsert_faostat_records
from app.warehouse.db import get_engine


def _build_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--item", required=True, nargs="+", help='real FAOSTAT Item name(s), e.g. "Poppy seed"'
    )
    parser.add_argument(
        "--zip-path",
        default=None,
        help="use an already-downloaded zip instead of fetching a fresh one",
    )
    return parser.parse_args(argv)


async def main() -> None:
    args = _build_args()
    item_names = set(args.item)

    if args.zip_path:
        zip_path = args.zip_path
    else:
        print(f"downloading {PRODUCTION_ZIP_URL} ...", flush=True)
        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
            zip_path = tmp.name
            async with (
                httpx.AsyncClient(timeout=120.0) as client,
                client.stream("GET", PRODUCTION_ZIP_URL) as response,
            ):
                response.raise_for_status()
                async for chunk in response.aiter_bytes():
                    tmp.write(chunk)
        print(f"downloaded to {zip_path}", flush=True)

    records = load_production_zip(zip_path, item_names=item_names)
    print(f"parsed {len(records)} real records for items={sorted(item_names)}")

    engine = get_engine()
    written = await upsert_faostat_records(engine, records)
    print(f"written: {written}")
    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
