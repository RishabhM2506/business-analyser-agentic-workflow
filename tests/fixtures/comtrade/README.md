# Comtrade fixtures — provenance

No UN Comtrade API key exists yet (the user is providing one separately —
see `docs/OPEN-QUESTIONS.md`), so `tests/integration/test_comtrade_client.py`
never makes a live or authenticated call. These fixtures stand in for real
upstream responses.

- `horses_import_2022.json` — **verbatim, byte-for-byte** capture of a real
  live response from the no-auth preview endpoint, 2026-08-16:
  `GET https://comtradeapi.un.org/public/v1/preview/C/A/HS?reporterCode=699&period=2022&cmdCode=010121&flowCode=M`.
  Chosen because it happens to contain one aggregate/provisional row
  (`partnerCode=0`, "World", `isReported=false`, `isAggregate=true`) and two
  genuinely-reported partner rows (`partnerCode=826` United Kingdom,
  `partnerCode=36` Australia, both `isReported=true`, `isAggregate=false`)
  in a single response — good coverage of both completeness-flag branches
  from one real fixture.

- `tea_export_2023_partial.json` — **reconstructed from real values**, not a
  byte-for-byte capture. The live response for
  `GET .../preview/C/A/HS?reporterCode=699&period=2023&cmdCode=090240&flowCode=X`
  (2026-08-16) had 258 records and exceeded the fetch tool's output budget;
  the first 15 records' real field values (`partnerCode`, `primaryValue`,
  `qty`, `qtyUnitCode`, `isReported`, `isAggregate`) were captured, then
  assembled into a full envelope using the exact schema verified byte-for-byte
  in `horses_import_2022.json` (the constant-across-records fields —
  `typeCode`, `freqCode`, `classificationCode`, etc. — verified identical in
  both real captures). Used for the "many partners in one response" shape,
  where every row happens to be `isReported=false` (a real, useful data
  point: recent-year per-partner detail is often still modeled/estimated —
  see `docs/PHASE0-FINDINGS.md` §4).

- `empty.json` — a legitimate empty-result envelope
  (`{"count": 0, "data": [], "error": ""}`), matching the verified envelope
  shape with zero records — a real, observed shape (a narrow query during
  live verification returned exactly this), not fabricated.

- `error_envelope.json`, `malformed.json` — synthetic, for the error/schema-
  validation paths that can't be captured from a healthy live call by
  construction (Comtrade only documents the `error` field's existence, not
  a sample non-empty value) — clearly not real recordings.
