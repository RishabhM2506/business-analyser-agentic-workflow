"""Real, curator-facing CLI: loads one downloaded BACI ZIP into
`raw_baci_records` for a set of tracked HS6 codes. BACI has no API — the
ZIP is a manual download from `https://www.cepii.fr/DATA_DOWNLOAD/baci/doc/baci_webpage.html`
(real current vintage/URLs verified live, `docs/BUILD-LOG.md`); this
script is the load step only, not a fetcher.

Usage (from the repo root, with `DATABASE_URL` pointed at a real
Postgres):

    uv run python scripts/load_baci_zip.py \\
      --zip-path /path/to/BACI_HS22_V202601.zip \\
      --vintage 202601 --hs-revision 22 --years 2022 2023 2024 \\
      --hs6 120791

`--vintage`/`--hs-revision` must match the ZIP's own real filename
convention (`BACI_HS<revision>_Y<year>_V<vintage>.csv` per member) — the
loader raises rather than guessing if a requested year's member isn't
found in the ZIP.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.pipeline.baci import load_baci_zip, upsert_baci_records
from app.warehouse.db import get_engine


def _build_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--zip-path", required=True)
    parser.add_argument("--vintage", required=True, help='e.g. "202601" (no "V" prefix)')
    parser.add_argument("--hs-revision", required=True, help='e.g. "22" for the HS 2022 revision')
    parser.add_argument("--years", required=True, type=int, nargs="+", help="e.g. 2022 2023 2024")
    parser.add_argument("--hs6", required=True, nargs="+", help="tracked HS6 codes, e.g. 120791")
    return parser.parse_args(argv)


async def main() -> None:
    args = _build_args()
    records = load_baci_zip(
        args.zip_path,
        vintage=args.vintage,
        hs_revision=args.hs_revision,
        years=args.years,
        hs6_codes=set(args.hs6),
    )
    print(f"parsed {len(records)} real India-involving records")

    engine = get_engine()
    written = await upsert_baci_records(engine, records)
    print(f"written: {written}")
    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
