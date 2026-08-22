# REVIEW — PM, business analysis (Step 2, iteration 2)

Re-reading `docs/PLAN.md` against iteration 1's findings. All 9 are genuinely fixed, not just
acknowledged — I checked each one specifically:

- Check A now has a real query (India as reporter, §8 Query 1) — verified the join in §10 actually reads
  `reporter_code='699'` for check A and `partner_code='699'` for check B, matching the two query shapes.
- `ref_country_crosswalk` exists with an explicit unmapped-name policy (routes to "all other partners" +
  dead-letters, never drops or misjoins).
- `analytics_partner_rankings` is re-keyed on partner, not rank — a partner's row now persists every year
  with a real status even with no data that year. This is the fix I most wanted to see, and it's correct.
- Check B's partner dimension is in both the DDL and the facts JSON now.
- `ref_hs6_hs8_crosswalk` is a real, derived-not-guessed mechanism.
- The mismatch join path is named and owned (`report/mismatch.py:compute_checks`).
- Landed cost moved to monthly grain — mid-year duty changes are no longer ambiguous.
- ITC-HS/international-HS6 divergence has an explicit assumption + escape-hatch table.
- D12 now has a real structural gate (`regulatory_note_missing_warning`), not just a hopeful data model.
- The quintal→kg conversion is named and placed at the normalizer boundary.

I don't have five new findings at this depth, and I'm not going to invent filler to hit a number — the
gate is genuinely closer to passable than iteration 1. But re-reading the fixes themselves (not just
whether they exist) surfaced three real follow-on gaps the fixes introduced or left open:

## Findings

### MAJOR — "All other partners" aggregation isn't restated for the new nullable-rank schema
§12 still says: "'All other partners' row = Σ of every rank beyond top_n... status = OK if every
constituent is OK, else the worst status present among constituents." But `analytics_partner_rankings` now
routinely contains partners with `rank=NULL` and `value_inr_paise=NULL` (the exact fix from iteration 1) —
"every rank beyond top_n" doesn't obviously include or exclude them, since they have no rank to compare
against `top_n` at all. If the sum silently only includes non-null-rank rows beyond top_n, that's fine
arithmetically, but the "all other partners" total would then not actually equal "total minus top-N" if
there's a `FETCH_FAILED` partner missing from it — and the report claims reconciling totals as one of its
core honesty guarantees. §12 needs one more sentence: null-rank partners' values are definitionally absent
from the sum (there's nothing to sum), but their *status* still needs to surface somewhere in "all other
partners" (e.g. its own status is `OK` only if every ranked-beyond-top-n constituent is OK *and* there are
no null-rank partners in that same group hiding a fetch failure — otherwise the aggregate status must
reflect the worst case, same principle as before, just needs restating against the new schema).

### MINOR — 60% HHI threshold for `regulatory_note_missing_warning` is invented, not verified
Every other numeric threshold in this plan (D9's 15%/40% mismatch bands, D11's 30% coverage floor) comes
from the master prompt itself. The 60% top-1-partner-share trigger I added in iteration 2 is mine, with no
empirical basis — reasonable as a starting point, but it should be flagged the same honest way §1 flags
other unverified numbers, not presented with the same confidence as the prompt's own non-negotiable bands.

### MINOR — Partner-universe backfill-before-first-appearance isn't stated either way
If a country starts trading poppy seed with India for the first time in 2023, does
`analytics_partner_rankings` get a row for that country in 2021/2022 (status presumably `NOT_REPORTED`,
which would be misleading — they weren't "not reported," trade with them for this product genuinely didn't
exist yet), or does the partner's row history start exactly at first appearance? The fix description implies
the latter ("once a partner is seen once, it is never removed, only ever gains new yearly rows") but never
explicitly rules out backfilling earlier years, which would be the wrong call if implemented that way. One
sentence closes this: no backfill before first appearance, full stop.

## What I checked and did not flag

Re-verified the two-query Comtrade design doesn't quietly double the retry/rate-limit surface without
accounting for it — §8 explicitly says the limiter and circuit breaker are shared across both query shapes,
which is correct (same underlying Comtrade quota either way). Re-checked that `ref_hs6_hs8_crosswalk` being
DGCIS-scrape-derived rather than manually maintained doesn't create a chicken-and-egg problem for the
canonical scenario's own first check ("does 120791 split") — it doesn't, since the very first scrape run
for that hs6 populates it before any report is generated. No further findings there.

VERDICT: CHANGES_REQUESTED
