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
