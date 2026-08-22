# REVIEW — PM, business analysis (Step 2, iteration 1)

I'm reading this as the person who will use this report to decide whether to import poppy seeds. I don't
care about the elegance of the schema. I care whether the number on my screen is honest. Reading
`docs/PLAN.md` with that lens, several places where the plan *describes* an invariant but doesn't actually
*enforce* it structurally.

## Findings

### BLOCKER — Mismatch check A cannot be computed from the query design in §8
D9 defines check A as "DGCIS vs India's own Comtrade submission" — i.e. it needs India's own reported
trade (`reporterCode=699`) as a Comtrade *reporter*. But §8's Comtrade request template fixes
`partnerCode=699` with `reporterCode` omitted/all — that query only ever returns *other countries'*
reported trade with India (India as partner). It never fetches India's own Comtrade submission. As written,
there is no query anywhere in this plan that produces the data check A needs. Either §8 needs a second
query shape (`reporterCode=699`, `partnerCode=all`), or check A needs to be re-scoped and its formula
corrected — right now it's a formula referencing data the ingestion design never fetches.

### BLOCKER — No country-code crosswalk between DGCIS's free-text names and Comtrade/BACI's numeric codes
`normalized_trade_flows.partner_country_code` is described as "normalized to a shared country code list,"
but `raw_dgcis_monthly.partner_country` is a raw scraped string, while Comtrade/BACI use numeric
codes. There is no reference table, no fuzzy-matching policy, no manual-override mechanism described
anywhere. Every single mismatch check (D9) and every partner-level ranking (D14) depends on correctly
joining "the same country" across three differently-coded sources. A silent misjoin here (e.g. DGCIS's
"Turkiye" vs Comtrade's "Turkey" vs a stale ISO code) would silently corrupt every downstream number with
no error, no `FETCH_FAILED`, nothing — the exact "actively misleading" failure mode Step 2 is supposed to
catch. This needs its own reference table and an explicit unmatched-name policy (does an unmatched DGCIS
country name become its own `NOT_TRACKED`-style bucket, or silently drop into "all other partners"? Either
is defensible, but the plan must say which).

### BLOCKER — "Does a partner disappear because they stopped trading, or because the pipeline broke?" is not actually answerable from this schema
This is my own headline question from the prompt, and I went looking for the answer in
`analytics_partner_rankings`. It's not there. The table's PK is `(hs6, flow, year, rank)` — nothing says
whether it's populated with **every partner that has ever appeared for this hs6/flow**, with an explicit
`NOT_REPORTED`/`ZERO` row for years they didn't trade, or whether a partner with no data for a given year
simply has **no row at all** for that year. If it's the latter (which is what "precomputed ranked list"
naturally reads as — you can't rank a country with a null value), then a partner that stopped trading is
*indistinguishable* from a partner our scraper failed to fetch: both just vanish from the list. This is
D1/D3's entire purpose, failing on the single most obvious PM-facing case. The plan needs to say explicitly:
every partner that has ever appeared for this hs6 gets a row every year, with a real status, forever.

### BLOCKER — Mismatch check B has no partner dimension in either the DDL or the facts JSON
D9 defines check B as inherently per-partner ("DGCIS import vs **partner's** Comtrade export"). But
`analytics_mismatch_checks`'s primary key is `(hs6, flow, year, check_name)` — no `partner_country_code`
column — and the facts JSON example for a mismatch entry (`{"check": "B_...", "year": 2025, "gap_pct": 9.1,
"severity": "quiet"}`) also has no partner field. As written, there is nowhere to store "Turkey's gap is
9%, but Country X's gap is 55%" simultaneously for the same year — the schema can hold exactly one B-check
result per (hs6, flow, year), which can't be what's intended given the canonical scenario explicitly wants
"any partner outside the 5–12% band flagged with severity" (plural, per-partner). This is a real structural
bug, not a nuance — the primary key needs `partner_country_code` added.

### MAJOR — HS6→HS8 resolution mechanism is asserted, never designed
The canonical scenario's first required check is "does 120791 split into multiple ITC-HS8 lines" — and the
facts JSON has an `hs8_split_note` field ready to hold the answer — but nowhere in the plan is there a
described mechanism for *how* we know which HS8 lines sit under a given HS6. Is DGCIS's own commodity search
queryable by HS6 and does it return every HS8 child automatically? Is there a separate maintained crosswalk
table (like `ref_duty_rates`)? Given the plan's own verified finding that ITC-HS lines are actively being
"dropped or re-allocated... from April 2026," this crosswalk needs to be a first-class, versioned reference
table, not implied. Right now this is the one part of the canonical scenario with literally no design
behind it.

### MAJOR — No described join path from `normalized_trade_flows` to `analytics_mismatch_checks`
§10 gives formulas for check A/B/C in terms of "dgcis_total" / "comtrade_india_reported_total" / etc, but
`analytics_mismatch_checks` is populated by *something* that has to join three different `source` values
of `normalized_trade_flows` on `(hs6, flow, year[, partner])` — and that join logic (which query, which
module, run when) isn't named anywhere in §3's module layout. `report/mismatch.py` is listed but its actual
inputs aren't specified. Given finding #2 above (no country-code crosswalk), this join is exactly where a
silent bug would hide.

### MAJOR — Mid-year duty-rate changes have no resolution rule
`ref_duty_rates` correctly supports `effective_from`/`effective_to` date ranges, meaning a rate can change
mid-fiscal-year. But `analytics_landed_cost`'s PK is `(hs8, year)` — one row per calendar year. If BCD
changes in February, which rate populates that year's single row — the rate at Jan 1, the rate as of the
report's `data_as_of` date, a weighted average, or the latest rate applicable within the year? The plan
needs to pick one and say so; right now it's silently ambiguous, and a Budget-day rate change would produce
a landed-cost figure nobody could reproduce or explain.

### MAJOR — ITC-HS vs international HS6 divergence is not addressed
DGCIS reports against India's own ITC-HS nomenclature, not literally the international HS used by
Comtrade/BACI. These usually agree at the 6-digit level but are not guaranteed to, especially around a
revision event (which the plan itself found is happening right now, April 2026). `hs_revision` is a column,
but there's no described reconciliation step for the case where ITC-HS's HS6 grouping doesn't match
international HS6 for some code. Worth at minimum a documented assumption ("we treat ITC-HS8's first 6
digits as equivalent to international HS6 and flag exceptions X, Y" or similar) rather than silence.

### MAJOR — D12's regulatory-note requirement has data model but no enforcement
`ref_regulatory_notes` exists and feeds the facts JSON, but nothing in the plan *requires* it to be
populated before a report is generated, or gates/flags a report for an HS6 with no regulatory note when
trade concentration looks suspiciously regulated (D9/HHI would actually be a good signal for this). D12's
own stated failure mode — "the narrative will explain a licensing regime as a commercial preference" — has
no described guard against exactly that happening for any HS6 the curator hasn't gotten to yet. This needs
either a hard gate (no regulatory note → narrative omits any commercial-preference framing entirely, stated
explicitly) or an explicit accepted-risk note.

### MINOR — Agmarknet's per-quintal unit is never converted to per-kg in any named formula
`raw_agmarknet_prices.modal_price_inr_paise_per_qtl` is quintal-denominated (100 kg); `analytics_landed_cost.domestic_price_inr_paise_per_kg`
is per-kg. §11's margin formula doesn't show the ÷100 conversion step, and no module in §3 is named as
owning it. Low severity only because it's a mechanical, obviously-necessary conversion once someone writes
the code — but a 100x error here would silently produce a nonsense margin, so it's worth naming explicitly
rather than assuming whoever implements `landed_cost.py` remembers it.

## What I checked and did not flag

The FX/D8 section is genuinely solid — the live verification of Frankfurter's actual weekend behavior
(finding it contradicts the prompt's own stated assumption) is exactly the kind of thing that would have
produced a silently-wrong "which date was used" UI note if it had gone unchecked. No further FX findings.
The status-enum-to-source mapping (§5) is concrete and testable as written. The D5/D6 Comtrade retry design
is fully specified modulo the honestly-flagged live-batching unknown, which is appropriately deferred to
Step 3 rather than guessed.

VERDICT: CHANGES_REQUESTED
