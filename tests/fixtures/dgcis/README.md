# DGCIS fixtures — provenance

- `poppy_seed_turkey_import_annual.html` — **verbatim, byte-for-byte** capture of a real live response,
  2026-08-23:
  `POST https://tradestat.commerce.gov.in/eidb/commodityx_countries_wise_import`
  with `searchTerm=12079100&ContEidbi=409&ContEidbyi=2024&ReportEidbi=1` (HS6 120791 poppy seeds, Turkey,
  FY2024-25, values in ₹ Crore), after a matching `GET` to the same URL for a fresh CSRF token and session
  cookie (`docs/PLAN.md` §1's verified mechanics).

  Contains a real 5-year annual series (FY2020-21 through FY2024-25) for one (country, HS8) pair — the
  report type `docs/PLAN.md` §1/§7 settled on for the annual, per-partner ingestion path. Real values:
  `4.91, 0.00, 424.66, 0.00, 0.00` (₹ Crore) — genuinely volatile, consistent with India's real, on-again/
  off-again restrictions on Turkish poppy-seed imports over narcotic-content compliance (not scraper noise).
  Also carries the reported unit ("KGS") and the "ITC HS Code... dropped or re-allocated... from April
  2026" CODE_RETIRED-relevant footnote elsewhere on the page, both real.

  Parser tests (`tests/unit/pipeline/test_dgcis_parser.py`) run against this fixture, not a live call.

- `poppy_seed_monthly_import_jun2022_value.html` / `_quantity.html` — **verbatim** captures of real live
  responses, 2026-08-23, `POST https://tradestat.commerce.gov.in/meidb/commoditywise_import` for HS8
  `12079100`, June 2022, `imddReportVal=3` (₹ Crore) and `=2` (quantity) respectively — the two real,
  separate requests `fetch_monthly_record` combines into one record (no single request returns both).
  Real values: `166.50` ₹ Crore, `6,347,970` KGS, both marked `"(R)"` (Revised/Final — a fully finalized
  past month).

- `poppy_seed_monthly_import_jun2026_flash.html` — real capture, June 2026 (₹ Crore), marked `"(F)"` —
  a recent, published-but-not-yet-finalized month (real value `0.00` for this specific product, but the
  page's own "India's Total Import" footer row shows a real nonzero national total, confirming June 2026
  data genuinely exists and simply has zero real poppy-seed trade that month).

- `poppy_seed_monthly_import_aug2026_not_yet_published.html` — real capture, August 2026 (the literal
  current month at capture time), marked `"(A)"` (Advance) — genuinely unpublished: *both* the specific
  commodity's value and the "India's Total Import" national-total footer row read `0.00`, confirming the
  whole month's collection hasn't happened yet, not a coincidental real zero for poppy seeds specifically.

  These four monthly fixtures together verify all three of DGCIS's real revision-status markers
  (`docs/PLAN.md` §1, D15) — `parse_monthly_response` tests (`tests/unit/test_dgcis_parser.py`) run
  against them, not live calls.
