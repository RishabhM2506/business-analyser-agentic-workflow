# PLAN — India Trade Analysis Pipeline (DGCIS-backbone)

Status: APPROVED (Step 2, iteration 3) as the Step 3 build reference, with one user-directed amendment
since approval: §4a (evidence-first duty data), added 2026-08-23 in response to explicit user direction —
not re-run through the Step 2 gate since it strengthens an already-approved invariant (D1/D2's "never a
dash, always a reason") rather than changing anything the PM already reviewed. Earlier revision history:
`docs/REVIEW-PM.md` iteration 1 (4 BLOCKER + 4 MAJOR + 1 MINOR, marked **[PM-1 fix: ...]**) and iteration 2
(1 MAJOR + 2 MINOR follow-on findings, marked **[PM-2 fix: ...]**), both addressed inline at the point of
the fix rather than in a separate changelog.

## 0. Repo survey — what already exists here

This pipeline is being built **inside** `business-analyser-agentic-workflow`, as a **new, standalone
capability alongside** the existing product (free-text search → HS6 → single-Comtrade-call trade table,
live at `POST /threads/{id}/messages` and `POST /threads/{id}/search`). It does not replace that feature.

**Conventions found and being followed:**
- Python 3.12, `uv` for deps/lockfile, FastAPI, `pydantic`/`pydantic-settings` v2, `structlog`, `httpx`,
  `tenacity` for retry, `mypy --strict`, `ruff`, `black`, `pytest` (`unit`/`integration` markers).
- Settings via `app/settings.py` (`pydantic-settings`, env-driven, no hardcoded credentials) — extend, don't
  duplicate.
- Reference data committed under `data/*.csv` (e.g. `harmonized-system.csv`, `comtrade-partner-areas.csv`)
  and loaded via an `lru_cache`d provider function — the established pattern for static lookups
  (`app/knowledge/provider.py`).
- `app/tools/comtrade_client.py` already exists: real, live-verified request shape
  (`GET {base}/data/v1/get/C/A/HS`, params `reporterCode`/`period`/`cmdCode`/`flowCode`), JSON envelope
  (`{data: [...], error: ...}`), 429/5xx retryable via `tenacity.AsyncRetrying` +
  `wait_exponential_jitter`. **Not reused as-is**: it's built for one interactive lookup (one year, one
  flow, one reporter) with exponential backoff suited to a request path. This pipeline's nightly Comtrade
  mirror job needs a different retry contract (fixed 30/60/120/300s schedule, D6) and a bulk multi-value
  request shape (D5) — a new client, in `app/pipeline/comtrade_mirror.py`, reusing only `httpx`/`tenacity`.
- **Product-name → HS6 resolution already exists and is reused, not rebuilt**: `app.search.service.search_products`
  (BM25 + vector + LLM rerank, live, tested this session) is the front door. This plan's new capability
  starts from "HS6 code known" — `POST /threads/{id}/trade-report`, a new, deeper, precomputed report,
  alongside (not replacing) the existing lightweight `/messages` flow.
- **No existing domain-data persistence.** Postgres is present in `docker-compose.yml` but used *only* as
  LangGraph's checkpoint store (`langgraph-checkpoint-postgres`) — no application schema, no migration
  tool (no Alembic), no ORM models for business data. This pipeline is the first thing in this repo to need
  real persistent domain tables. New: Alembic for migrations (`migrations/`, `alembic.ini` at repo root,
  matching the existing top-level layout of `data/`, `scripts/`, `prompts/`), SQLAlchemy Core `Table`
  objects (not ORM classes — bulk-COPY-friendly, matches D7/D-perf's bulk-upsert requirement) in
  `app/warehouse/schema.py`.
- **No Redis anywhere in this stack.** New dependency (`redis`, async client) and a new `redis` service in
  `docker-compose.yml`, required by D8's FX cache.
- Docker: `Dockerfile` only copies `app/`, `prompts/`, `data/` — the new `app/pipeline/`, `app/warehouse/`,
  `app/fx/`, `app/report/` packages land under `app/` so no Dockerfile change is needed for code; the new
  `migrations/` directory **does** need adding to the Dockerfile copy list (for running migrations in the
  deployed container).

## 1. Verified external facts (live-checked 2026-08-23, not assumed)

**FX (Frankfurter) — verified, resolves the open item in D8:**
- Latest: `GET https://api.frankfurter.dev/v2/rate/USD/INR` → `{"date":"2026-08-22","base":"USD","quote":"INR","rate":95.67}` — matches the prompt's claimed shape exactly.
- Historical: `GET https://api.frankfurter.dev/v2/rate/USD/INR?date=YYYY-MM-DD` (query param, **not** a path
  segment — `/v2/{date}/rate/...` 404s). Verified: `?date=2021-06-15` → `rate: 73.349`.
- Range/time-series: `GET https://api.frankfurter.dev/v2/rates?from=YYYY-MM-DD&to=YYYY-MM-DD&base=USD&quotes=INR`
  → an array, one entry per calendar date in range.
- **Real, verified deviation from the prompt's D8 assumption**: I tested Fri 2024-01-05 / Sat 01-06 / Sun
  01-07 and Christmas Day 2023-12-25 explicitly — **every calendar date returns a distinct, non-repeated
  rate value.** There is no gap on weekends/holidays and no flag distinguishing a "real" business-day rate
  from an interpolated one. This directly contradicts D8's "weekends and holidays have no published rate —
  carry forward the last published rate and record which date was actually used": **there is nothing to
  carry forward — Frankfurter v2 apparently interpolates/publishes for every day now.**
  - **Design consequence**: implement the carry-forward code path anyway (defensive — for the one real
    failure mode that *does* exist, Frankfurter being unreachable), but do not build weekend-gap-detection
    logic expecting to find gaps, and do not claim a "which date was actually used" UI note for weekends —
    there is no such distinction to surface for this API version. `FX_STALE` will in practice only ever
    fire from a genuine API failure, not a weekend.
- Cache design (D8) stands as specified: Redis key `fx:USD:INR:<YYYY-MM-DD>`, historical = no expiry,
  today = expire at end of day IST, one call/day max, fallback to most recent cached value + `FX_STALE` +
  the cached rate's own date on failure.

**DGCIS Tradestat — verified live, Step 3 (2026-08-23), full mechanics confirmed and one major structural
finding that changes the ingestion design:**

- **Real request mechanics, fully verified with live GET+POST round-trips**: a Laravel application.
  `GET` a report page → parse the CSRF token out of `<input name="_token">` and capture the
  `Set-Cookie: indiatrade-session=...` (15-minute `Max-Age`, `httponly`, `secure`, `samesite=lax`) →
  `POST` back to the *same URL* with the token, the session cookie, and the form fields, within that
  15-minute window. Verified this actually works end to end: a real POST for HS `12079100` (poppy seeds)
  returned a real table containing "POPPY SEEDS W/N BROKEN" and real figures.
- **Field names are not standardized across report pages — verified by finding the actual bug live**: the
  visible "Enter HS Code" text input (`name="hscode_value"`) is **not** the field the form actually
  submits — it belongs to a separate "Search HSCode" lookup-assist modal. The real field is a different,
  initially-`disabled` input tied to the "specific commodity" radio button, and its name differs *per
  report page* (`Eidb_hscodeCwi` on one report, `comval` on another, `searchTerm` on a third — all verified
  live). Every report page needs its own one-time field-name verification; there is no shared convention
  to assume.
- **Two parallel sections exist**: `/eidb/*` (annual figures, fiscal year only — confirming the "coverage
  is FY not CY" finding below) and `/meidb/*` ("Monthly Export Import Data Bank" — real monthly figures,
  and its `ddReportYear`/`imddReportYear` select offers **"Financial Year" or "Calendar Year" directly**,
  verified live with a real Calendar Year request. This resolves the FY/CY ambiguity flagged below: request
  in Calendar Year mode and DGCIS itself does the FY→CY reframing, no client-side conversion needed.
- **Major finding — no report returns a commodity×country cross-tab in one call, despite report names
  suggesting otherwise**: exhaustively checked every plausible candidate live:
  - `eidb/commodity_wise_import` / `meidb/commoditywise_import` ("commodity-wise"): a **national total**,
    two-period time comparison (e.g. Jun-2023 vs Jun-2024 + %growth) — no country dimension at all.
  - `meidb/country_wise_import`: the mirror image — **one country's total across all commodities**, no
    commodity dimension at all (verified live for Turkey, June 2024: table header is just "Country", one
    row, no per-commodity breakdown).
  - `eidb/commodityx_countries_wise_import` (the name that most strongly suggested "one commodity, all
    countries"): its country field (`ContEidbe`, 251 real country options, verified) is **`required` with
    no "all countries" option** — this report also returns exactly one country per request, despite the
    name.
  - `meidb/country_wise_all_commodities_export/import`: same pattern — its country field
    (`cwcexallcount`, 252 options) also has no "all" choice.
  - **Conclusion**: getting "HS8 code X, broken down by every partner country" requires looping over
    India's ~250 real trading-partner country codes (a bounded, enumerable, already-captured list, real
    codes verified: e.g. Turkey=409) — not the single clean call originally hoped for. This is a real,
    structural fact about this government portal, not a scraper design shortcut to fix later.
  - **Good news, verified after closing the "no cross-tab" question**: `commodityx_countries_wise_import`
    (real fields verified: `searchTerm`=HS8 code, `ContEidbi`=country code, `ContEidbyi`=year,
    `ReportEidbi`=value-type) returns a **5-year annual time series in one response**, not one year — a
    real POST for HS `12079100` x Turkey returned FY2020-21 through FY2024-25 side by side in a single
    table, plus the reported unit ("KGS") directly in the response header (feeds D10's unit-consistency
    check for free). This meaningfully revises the volume estimate down: **~250 requests total per tracked
    HS8 code** (one per country, each yielding 5 years of annual data), not ~250 *per period*. **Ingestion
    job design consequence (§7)**: still real, non-trivial volume for a portal with no documented rate
    limit — needs its own rate-limit/backoff design — but far more tractable than initially feared, and an
    annual refresh (not monthly) can cover the full window in one pass per country per tracked code.
  - Real data sanity check: Turkey/poppy-seed values across the 5 years were `4.91, 0.00, 424.66, 0.00,
    0.00` (₹ Crore) — a genuinely volatile, non-monotonic real pattern, consistent with India's actual,
    real on-again/off-again restrictions on Turkish poppy-seed imports over narcotic-content compliance
    (a real regulatory story, not scraper noise) — exactly the kind of pattern D12's regulatory-note field
    exists to explain.
  - **Resolved, and cleanly**: checked whether `meidb` has a `commodityx_countries_wise_import` sibling —
    it does not (`404`, confirmed live). **DGCIS's per-partner-country breakdown is only available at
    annual granularity; monthly data is only available as a national total (no country dimension).** Read
    again against what the two report sections actually need (not what `raw_dgcis_monthly`'s schema
    assumed): the canonical scenario's "annual import series... by partner" section and D15's "current-year
    month-wise" section were never meant to share one partner-broken-down monthly table — D15 only asks for
    a month-by-month *trend*, which is exactly what the national-total monthly report already is. The two
    real DGCIS report types map cleanly onto the two real report sections:
    - `commodityx_countries_wise_import`/`_export` (annual, per-partner, 5-year batches) → the annual,
      by-partner series (§12/D14).
    - `commoditywise_import` (monthly, national total only) → D15's month-wise section.
    `raw_dgcis_monthly`'s `partner_country` column needs one addition to reflect this: a sentinel value
    (`'ALL_PARTNERS'`, matching the sentinel already used in `analytics_mismatch_checks`, §4) for rows
    sourced from the monthly national-total report, since that source genuinely has no partner dimension to
    record — not a gap in what was scraped, a property of what DGCIS itself publishes at that granularity.
  - **Canonical scenario's first "verify in week one" check, resolved live**: does `120791` split into
    multiple ITC-HS8 lines? **No.** Searching `searchTerm=120791` (bare HS6) and `searchTerm=12079100`
    (the specific HS8) against the same country/year returned byte-for-byte identical values
    (`4.91, 0.00, 424.66, 0.00, 0.00`) — the same single commodity master record either way. `12079100` is
    the only ITC-HS8 line beneath `120791`, confirmed, not assumed — the `hs8_split_note` in §14's facts
    JSON is a real, verified fact for this product, not a placeholder. (Also noted: searching by bare HS6
    doesn't *resolve* to the canonical 8-digit code, it just echoes back whatever digit-length was
    searched, and drops the `Unit` field — real API behavior worth remembering, not relied on as a
    discovery mechanism for HS6→HS8 in general; the "HS Code Search" lookup modal, `hscode_fetch`/
    `description_value`, found earlier but not yet explored, is the more likely real discovery path for a
    tracked HS6 with genuinely multiple HS8 children.) `ref_hs6_hs8_crosswalk` has one real, live-verified
    row: `(hs6='120791', hs8='12079100')`.
- **Coverage is stated as Indian *fiscal* years ("2017-2018 to 2025-2026"), not calendar years** — the
  prompt's "Jan 2018 onward" is a simplification; FY2017-18 starts April 2017. `meidb`'s year dropdowns
  (verified live) actually start at 2018 (FY2018-19), one year later than the homepage's stated coverage —
  worth reconciling when the real ingestion job is built, not assumed either way.
- **Directly confirms D10/CODE_RETIRED is not hypothetical**: the live `eidb/commodity_wise_import`
  response itself renders the footnote "ITC HS Code of the Commodity is either dropped or re-allocated and
  the unit of the commodity may be changed from April 2026" — confirmed present in real rendered output,
  not just marketing copy.

**BACI (CEPII) — verified:**
- Direct ZIP download, no registration/login. Most recent vintage (`202601`, released Jan 2026) covers
  through 2024 — roughly a 13-month lag (prompt says "~18 months"; real lag appears shorter but still
  substantial and TBD-precise per vintage). HS revision is selectable (HS92 through HS22); we will pick the
  HS revision whose year range covers each report's window, since a single BACI vintage can span multiple
  HS revisions. Etalab 2.0 license (permits reuse). 200 countries, ~5,000 HS6 products, built from
  reconciled Comtrade data with CIF/FOB adjustment already applied by CEPII.

**Agmarknet / data.gov.in — NOT verified, real gap:**
- The catalog page returned HTTP 403 to an unauthenticated fetch (likely a bot-blocking response, not
  proof the API itself is unavailable). **I do not have a `data.gov.in` API key.** The real endpoint shape,
  resource ID for poppy seed / oilseed mandi prices, rate limits, and update cadence are all unverified.
  **This is a Step 6 credential request, and possibly earlier**: the Agmarknet ingestion job cannot be
  built against fixtures alone forever — a real key is needed before Step 3 can finish that job for real
  (fixture-based unit tests can still be written against a guessed-but-labeled-as-guessed response shape
  in the meantime, clearly marked as such).

**UN Comtrade bulk-batching (D5) — fully verified live, Step 3 (2026-08-23), using the real Comtrade key:**
- Every batching dimension D5 assumed works, confirmed with real calls against `GET /data/v1/get/C/A/HS`:
  - `period=2020,2021,2022,2023,2024` (5 years) — confirmed, all 5 periods returned in one response.
  - `flowCode=M,X` — confirmed, both flows returned in one response.
  - `reporterCode` omitted, `partnerCode=699` (Query 2, §8: every country's own reported trade with
    India) — confirmed, 34 distinct reporters returned in one call.
  - `partnerCode` omitted, `reporterCode=699` (Query 1, §8: India's own submission) — confirmed, multiple
    partner rows (including the `partnerCode=0` "World" aggregate) returned in one call.
  - Combining all of the above at once (`reporterCode=699`, `partnerCode` omitted, 5 comma-joined periods,
    `flowCode=M,X`) — confirmed in a single call, 265 rows returned.
  - **Bonus, not anticipated by D5's original framing**: `cmdCode` also accepts comma-separated values
    (`cmdCode=120791,090111` returned both codes' rows in one call) — meaning *multiple tracked HS6 codes*
    can batch into the same request too, not just periods/flows/reporters. Worth using once more than one
    HS6 is actively tracked — not required for the single-product canonical scenario, so not built into §8's
    request templates yet, but a real, verified option for later.
- No fallback needed — every dimension D5 required works exactly as designed. §8's two-query templates are
  now the confirmed, not just proposed, ingestion design.

## 2. Explicit non-goals

Per the prompt's "OUT OF SCOPE": no auth/accounts, no multi-tenancy, no shipment/company-level data, no
non-India-reporter world-vs-world analysis, no real-time/intraday data, no forecasting, no mobile app, no
billing. Additionally, specific to this plan: this pipeline does not touch or replace the existing
`/messages` (quick single-Comtrade-call) flow; duty rates are a **maintained reference table**, not scraped
or computed (no CBIC-tariff-scraping job in v1 — an operator updates `data/duty-rates.csv` per budget
change, same pattern as the existing `data/harmonized-system.csv`).

## 3. Module layout

```
app/
  pipeline/            # ingestion jobs — one file per source, each independently schedulable
    dgcis.py           # scrape + parse + normalize DGCIS monthly ITC-HS8 records
    comtrade_mirror.py # bulk nightly Comtrade pull (mirror/benchmark only)
    baci.py            # annual BACI ZIP download + load
    agmarknet.py        # daily mandi price pull
    duty_source.py       # DutySource Protocol + ManualDutySource (§4a) — evidence-first, cited
    dead_letter.py      # shared FETCH_FAILED write path + alerting hook (D3)
  warehouse/
    schema.py           # SQLAlchemy Core Table defs for raw_*/normalized_*/analytics_*/ref_*
    bulk_upsert.py       # COPY-based bulk upsert helpers (shared by all ingestion jobs)
    coverage_gate.py     # D11 — refuses metric computation below threshold
  fx/
    client.py            # Frankfurter client
    cache.py              # Redis-backed cache, D8 contract
    decomposition.py      # Δvalue ≈ Δqty × Δprice × ΔFX (D8 three-way split)
  report/
    mismatch.py            # D9 checks A/B/C
    unit_consistency.py    # D10 gate
    landed_cost.py          # consumes DutyEvidence (§4a), never raw percentages — partial-calc rule
    facts.py                 # assembles the frozen facts JSON (LLM contract)
    narrative.py               # LLM call + D4 post-validator
    service.py                  # orchestrates: coverage gate -> metrics -> facts -> narrative -> response
routes/
  trade_report.py               # POST /threads/{id}/trade-report (new endpoint, D14 params)
migrations/                     # new, Alembic
  versions/
alembic.ini                     # new, repo root
```

## 4. Schema DDL

All money columns are `BIGINT` **paise** (1 INR = 100 paise) — never `float`. Quantities are
`NUMERIC(18,3)`. FX rates are `NUMERIC(12,6)`. Every `normalized_*`/`analytics_*` row carries the full D7
lineage set.

```sql
-- ── Reference (maintained, not scraped) ──────────────────────────────────
-- [2026-08-23 revision, user-directed: "evidence-first" duty data — do not
-- fill a missing duty component with a guess, a secondary-source value, or
-- 0%.] Not a flat row of 4 percentages. One row per (hs8, component,
-- effective_from) — each component (BCD/AIDC/SWS/IGST) carries its own
-- verification status, citation, and dates independently, because a real
-- notification can update one component without touching the others, and
-- because "verified" is a per-fact property, not a per-row one.
--
-- NULL value_pct is structural, not incidental: a row's value_pct is only
-- ever non-NULL when verification_status is 'VERIFIED' or 'EXPIRED' — this
-- is what makes "NULL/unknown must never be interpreted as 0%" true by
-- construction rather than by convention a query could accidentally
-- violate. EXPIRED keeps its value_pct too (found via a real test, not
-- assumed): the user's own rule — "preserve it for historical analysis" —
-- requires the actual historical number, not just the fact one once
-- existed; only NOT_VERIFIED/CONFLICTING have no trustworthy single value.
-- landed_cost.py still excludes EXPIRED from any *current*/complete
-- calculation regardless, by checking verification_status='VERIFIED'
-- specifically, not "does a value exist."
CREATE TYPE duty_verification_status AS ENUM ('VERIFIED', 'NOT_VERIFIED', 'CONFLICTING', 'EXPIRED');

CREATE TABLE ref_duty_components (
  hs8                  TEXT NOT NULL,
  component            TEXT NOT NULL,        -- 'BCD' | 'AIDC' | 'SWS' | 'IGST'
  effective_from       DATE NOT NULL,
  effective_to         DATE,                  -- set (and status flipped to EXPIRED) when superseded
  verification_status  duty_verification_status NOT NULL,
  value_pct            NUMERIC(6,3),          -- NULL unless VERIFIED or EXPIRED
  source_authority      TEXT NOT NULL,        -- e.g. 'ICEGATE Trade Guide on Imports', 'CBIC Tax Information Portal'
  source_reference        TEXT NOT NULL,      -- notification/circular number, or 'none found' for NOT_VERIFIED
  source_url                TEXT,
  verified_date               DATE NOT NULL,  -- when a human curator last checked this, regardless of status
  notes                         TEXT,         -- conditions, caveats, why NOT_VERIFIED/CONFLICTING
  CHECK (component IN ('BCD','AIDC','SWS','IGST')),
  CHECK (
    (verification_status IN ('VERIFIED','EXPIRED') AND value_pct IS NOT NULL) OR
    (verification_status IN ('NOT_VERIFIED','CONFLICTING') AND value_pct IS NULL)
  ),
  PRIMARY KEY (hs8, component, effective_from)
);

-- Populated only for a component whose current row is CONFLICTING — the N
-- disagreeing official values found, each with its own citation. A
-- CONFLICTING row in ref_duty_components itself never carries a value_pct
-- (see the check constraint above); the candidates live here so nothing
-- forces an automatic pick between them (user's explicit rule: "do not
-- automatically choose one value").
CREATE TABLE ref_duty_component_conflicts (
  hs8              TEXT NOT NULL,
  component        TEXT NOT NULL,
  effective_from   DATE NOT NULL,
  candidate_value_pct NUMERIC(6,3) NOT NULL,
  source_authority     TEXT NOT NULL,
  source_reference       TEXT NOT NULL,
  source_url                TEXT,
  FOREIGN KEY (hs8, component, effective_from)
    REFERENCES ref_duty_components (hs8, component, effective_from)
);

CREATE TABLE ref_regulatory_notes (         -- D12
  hs6              TEXT PRIMARY KEY,
  note             TEXT NOT NULL,
  updated_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_by       TEXT NOT NULL
);

-- [PM-1 fix: BLOCKER "no country-code crosswalk"] DGCIS reports partner
-- country as a free-text string; Comtrade/BACI use numeric UN/ISO codes.
-- Every cross-source join (D9, D14 rankings) depends on this being correct
-- and explicit, not implicit. Maintained the same way as ref_duty_rates: a
-- committed CSV (data/country-crosswalk.csv), manually extended when a new
-- unmapped DGCIS name is seen (see dead_letter policy below).
CREATE TABLE ref_country_crosswalk (
  dgcis_country_name TEXT PRIMARY KEY,      -- exact string as DGCIS renders it
  country_code        TEXT NOT NULL,        -- UN M49 numeric code, matches Comtrade's reporter/partner codes
  updated_at            TIMESTAMPTZ NOT NULL DEFAULT now()
);
-- An DGCIS name with no crosswalk row does NOT block ingestion: the
-- normalizer writes partner_country_code = 'UNMAPPED' (never drops the
-- row), which routes into the "all other partners" bucket (D14) and is
-- also written to dead_letter_ingestion with source='dgcis_crosswalk_gap'
-- so it surfaces on the ops dashboard for a curator to add, rather than
-- silently misjoining against a wrong code or silently vanishing.

-- [PM-1 fix: MAJOR "HS6->HS8 mapping mechanism totally unspecified"] Not a
-- manually maintained table like duty rates — this is DERIVED from what
-- DGCIS's own scrape responses reveal. Every dgcis.py ingestion run
-- upserts every distinct (hs6, hs8) pair it actually observed that run;
-- "does 120791 split into multiple lines" becomes a direct query against
-- this table, not an assumption. effective_to is set (not deleted) when a
-- previously-seen hs8 stops appearing under its hs6 in a later run — this
-- is itself a CODE_RETIRED signal, feeding back into §5's status enum.
CREATE TABLE ref_hs6_hs8_crosswalk (
  hs6              TEXT NOT NULL,
  hs8              TEXT NOT NULL,
  first_seen_at     TIMESTAMPTZ NOT NULL,
  effective_to        TIMESTAMPTZ,          -- NULL = still observed in the most recent scrape
  PRIMARY KEY (hs6, hs8)
);

-- [PM-1 fix: MAJOR "ITC-HS vs international HS6 divergence not addressed"]
-- Documented assumption: ITC-HS8's leading 6 digits are treated as
-- equivalent to the international HS6 used by Comtrade/BACI/this repo's
-- own data/harmonized-system.csv taxonomy for join purposes. We do not
-- build automated description-similarity checking for v1 (real effort,
-- speculative payoff) — instead this is the explicit, human-populated
-- escape hatch for the rare case where that assumption is known to be
-- wrong for a specific code (most relevant right now given the live-found
-- April-2026 ITC-HS revision).
CREATE TABLE ref_hs_revision_notes (
  hs6              TEXT PRIMARY KEY,
  note             TEXT NOT NULL,           -- e.g. "ITC-HS8 12079100 does not map cleanly to intl HS6 120791 as of FY2026-27 revision; DGCIS totals for this code may be understated in check A/B/C"
  updated_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ── Raw layer: immutable, append-only, mirrors source shape ──────────────
-- [2026-08-23, corrected via real Step 3 investigation] DGCIS genuinely
-- publishes two different shapes (§1/§7) — this table is now specifically
-- for the MONTHLY, national-total-only report (meidb/commoditywise_import,
-- not yet built); partner_country here is always the 'ALL_PARTNERS'
-- sentinel (that source has no partner dimension to record). The
-- per-partner ANNUAL data lives in raw_dgcis_annual, below.
CREATE TABLE raw_dgcis_monthly (
  id               BIGSERIAL PRIMARY KEY,
  scraped_at       TIMESTAMPTZ NOT NULL,
  fiscal_year      TEXT NOT NULL,           -- "2024-2025" as DGCIS reports it
  calendar_month   DATE NOT NULL,           -- first-of-month, derived from fiscal_year + month label
  hs8              TEXT NOT NULL,
  flow             TEXT NOT NULL CHECK (flow IN ('import','export')),
  partner_country  TEXT NOT NULL,           -- 'ALL_PARTNERS' for this source (see note above)
  value_inr_paise  BIGINT,                  -- NULL if the source cell itself was blank/dash
  quantity         NUMERIC(18,3),
  unit             TEXT,
  raw_payload      JSONB NOT NULL,          -- full scraped row, for replay/debugging
  UNIQUE (fiscal_year, calendar_month, hs8, flow, partner_country)
);

-- Mirrors app.pipeline.dgcis.DgcisAnnualRecord exactly — one row per
-- (fiscal_year_label, hs8, flow, partner_country), real per-partner data
-- from commodityx_countries_wise_import/_export. No quantity column: this
-- report only ever returns value, never quantity (verified live) —
-- QTY_MISSING is the correct status for every row from this source, not
-- an ingestion gap.
CREATE TABLE raw_dgcis_annual (
  id                 BIGSERIAL PRIMARY KEY,
  scraped_at         TIMESTAMPTZ NOT NULL,
  fiscal_year_label  TEXT NOT NULL,          -- DGCIS's own label verbatim, e.g. "2020 - 2021"
  hs8                TEXT NOT NULL,
  flow               TEXT NOT NULL CHECK (flow IN ('import','export')),
  partner_country    TEXT NOT NULL,          -- DGCIS's own country name string, not yet normalized
  description        TEXT,
  unit               TEXT,
  value_inr_paise    BIGINT,                 -- NULL if the source cell itself was blank/unparseable
  raw_payload        JSONB NOT NULL,
  UNIQUE (fiscal_year_label, hs8, flow, partner_country)
);

CREATE TABLE raw_comtrade_records (
  id               BIGSERIAL PRIMARY KEY,
  fetched_at       TIMESTAMPTZ NOT NULL,
  period            INT NOT NULL,           -- year
  reporter_code     TEXT NOT NULL,
  partner_code      TEXT NOT NULL,          -- 699 = India, always, for this pipeline's calls
  flow_code         TEXT NOT NULL,
  cmd_code          TEXT NOT NULL,          -- hs6
  primary_value_usd NUMERIC(18,2),
  net_weight_kg     NUMERIC(18,3),
  is_reported       BOOLEAN NOT NULL,       -- false = row absent from response = NOT_REPORTED
  raw_payload       JSONB NOT NULL,
  UNIQUE (period, reporter_code, partner_code, flow_code, cmd_code)
);

CREATE TABLE raw_baci_records (
  id               BIGSERIAL PRIMARY KEY,
  loaded_at        TIMESTAMPTZ NOT NULL,
  vintage          TEXT NOT NULL,           -- e.g. "202601"
  hs_revision      TEXT NOT NULL,
  year             INT NOT NULL,
  exporter_code    TEXT NOT NULL,
  importer_code    TEXT NOT NULL,
  hs6              TEXT NOT NULL,
  value_fob_usd    NUMERIC(18,2),
  quantity_kg      NUMERIC(18,3),
  UNIQUE (vintage, year, exporter_code, importer_code, hs6)
);

CREATE TABLE raw_agmarknet_prices (
  id               BIGSERIAL PRIMARY KEY,
  fetched_at       TIMESTAMPTZ NOT NULL,
  price_date       DATE NOT NULL,
  commodity        TEXT NOT NULL,
  market            TEXT NOT NULL,          -- mandi name
  state             TEXT NOT NULL,
  modal_price_inr_paise_per_qtl BIGINT,     -- Agmarknet reports per-quintal
  raw_payload       JSONB NOT NULL,
  UNIQUE (price_date, commodity, market)
);

-- ── Dead letter (D3) ───────────────────────────────────────────────────
CREATE TABLE dead_letter_ingestion (
  id               BIGSERIAL PRIMARY KEY,
  source           TEXT NOT NULL,           -- 'dgcis' | 'comtrade' | 'baci' | 'agmarknet'
  job_run_id       UUID NOT NULL,
  attempted_at     TIMESTAMPTZ NOT NULL,
  request_desc     TEXT NOT NULL,           -- e.g. "hs8=12079100 flow=import fy=2024-2025"
  error_message    TEXT NOT NULL,
  attempt_count    INT NOT NULL,
  resolved         BOOLEAN NOT NULL DEFAULT false
);

-- ── Normalized layer: one canonical shape across sources ─────────────────
CREATE TYPE cell_status AS ENUM (
  'OK','ZERO','NOT_REPORTED','SUPPRESSED','NOT_YET_PUBLISHED','PROVISIONAL',
  'QTY_MISSING','UNIT_MISMATCH','CODE_RETIRED','FETCH_FAILED'
);

CREATE TABLE normalized_trade_flows (
  id                BIGSERIAL PRIMARY KEY,
  source            TEXT NOT NULL,          -- 'dgcis' | 'comtrade' | 'baci'
  hs6               TEXT NOT NULL,
  hs8               TEXT,                   -- NULL for sources that don't go below HS6 (Comtrade, BACI)
  hs_revision        TEXT NOT NULL,
  flow               TEXT NOT NULL CHECK (flow IN ('import','export')),
  period_month        DATE NOT NULL,        -- always first-of-month; annual sources use Jan 1 of that year
  calendar           TEXT NOT NULL CHECK (calendar IN ('CY','FY')),
  partner_country_code TEXT NOT NULL,       -- normalized to a shared country code list
  basis               TEXT NOT NULL CHECK (basis IN ('CIF','FOB')),
  currency             TEXT NOT NULL CHECK (currency IN ('INR','USD')),
  universe              TEXT NOT NULL,      -- e.g. 'india-customs', 'un-comtrade-mirror', 'baci-reconciled'
  dataset_version        TEXT NOT NULL,     -- source vintage/scrape-run id
  is_provisional          BOOLEAN NOT NULL,
  status                  cell_status NOT NULL,
  status_detail            TEXT,            -- e.g. expected publication date for NOT_YET_PUBLISHED
  value_inr_paise           BIGINT,         -- always populated in INR — DGCIS natively, others via FX (D8)
  value_original_currency_paise BIGINT,     -- the source's own currency, before any conversion
  fx_rate_used              NUMERIC(12,6),  -- NULL for DGCIS (never converted)
  fx_rate_date               DATE,
  quantity_kg                 NUMERIC(18,3),
  ingested_at                  TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE (source, hs6, hs8, flow, period_month, partner_country_code, dataset_version)
);

-- ── Analytics layer: precomputed on ingest, API reads only this ──────────
-- [PM-1 fix: BLOCKER "partner-disappeared vs pipeline-broke is unanswerable"]
-- Keyed on (hs6, flow, year, partner_country_code) — NOT (..., rank) — so a
-- partner that has EVER appeared for this hs6/flow gets a row for every
-- subsequent year, forever, even a year with zero data for them. `rank` is
-- nullable and populated only for rows with a real, comparable value
-- (status IN ('OK','ZERO')); a partner absent that year keeps its status
-- row (NOT_REPORTED / FETCH_FAILED / SUPPRESSED / etc, per §5) with
-- rank = NULL, so "this partner stopped trading" (NOT_REPORTED) and "our
-- scraper broke" (FETCH_FAILED) are always distinguishable, never a
-- silent gap in the list. The "partner universe" for a given (hs6, flow)
-- is maintained by the ingestion normalizer: once a partner is seen once,
-- it is never removed, only ever gains new yearly rows.
-- [PM-2 fix: MINOR "backfill-before-first-appearance ambiguous"] No
-- backfill: a partner's row history starts exactly at the year they first
-- appear for this (hs6, flow), never earlier. A year before a partner's
-- first appearance means "not yet a trading relationship for this
-- product," which is a different, true fact from NOT_REPORTED (which
-- means the relationship exists but that period's filing is absent) —
-- collapsing the two would misrepresent history, not just leave a gap.
CREATE TABLE analytics_partner_rankings (              -- D14 precompute strategy
  hs6              TEXT NOT NULL,
  flow             TEXT NOT NULL,
  year             INT NOT NULL,
  partner_country_code TEXT NOT NULL,
  rank                   INT,               -- NULL when status has no comparable value
  value_inr_paise           BIGINT,         -- NULL when rank is NULL
  status                     cell_status NOT NULL,
  PRIMARY KEY (hs6, flow, year, partner_country_code)
);
CREATE UNIQUE INDEX ix_apr_rank_where_present
  ON analytics_partner_rankings (hs6, flow, year, rank) WHERE rank IS NOT NULL;
  -- Slice to top-N at query time (D14): `WHERE rank <= :top_n AND rank IS NOT NULL`,
  -- never precomputed per (years, topN) combination.

CREATE TABLE analytics_monthly_current_year (            -- D15
  hs6              TEXT NOT NULL,
  flow             TEXT NOT NULL,
  month            DATE NOT NULL,           -- first-of-month, current CY only
  value_inr_paise  BIGINT,
  status            cell_status NOT NULL,
  status_detail      TEXT,
  is_provisional      BOOLEAN NOT NULL,
  mom_change_pct        NUMERIC(8,3),       -- month-on-month
  yoy_same_month_pct     NUMERIC(8,3),      -- vs same month last year
  data_as_of              TIMESTAMPTZ NOT NULL,
  PRIMARY KEY (hs6, flow, month)
);

CREATE TABLE analytics_unit_value_series (               -- D8 three-way decomposition
  hs6              TEXT NOT NULL,
  flow             TEXT NOT NULL,
  year             INT NOT NULL,
  unit_value_inr_paise_per_kg NUMERIC(18,4),
  delta_value_pct    NUMERIC(8,3),
  delta_from_qty_pct   NUMERIC(8,3),
  delta_from_price_pct  NUMERIC(8,3),
  delta_from_fx_pct      NUMERIC(8,3),
  coverage_gate_passed    BOOLEAN NOT NULL,  -- D11
  PRIMARY KEY (hs6, flow, year)
);

-- [PM-1 fix: BLOCKER "check B has no partner dimension"] check B is
-- inherently per-partner (D9: "DGCIS import vs partner's Comtrade
-- export"); checks A and C are aggregate (whole-hs6, not per-partner).
-- partner_country_code is part of the PK for all three so the schema is
-- uniform: A/C rows use the sentinel 'ALL_PARTNERS' (Postgres PKs can't be
-- NULL), B rows use the real partner_country_code — one row per partner
-- per year, so "Turkey is 9%, Country X is 55%" coexist correctly.
CREATE TABLE analytics_mismatch_checks (                  -- D9
  hs6              TEXT NOT NULL,
  flow             TEXT NOT NULL,
  year             INT NOT NULL,
  check_name        TEXT NOT NULL CHECK (check_name IN ('A_dgcis_vs_comtrade_india','B_dgcis_vs_partner_comtrade','C_dgcis_vs_baci')),
  partner_country_code TEXT NOT NULL DEFAULT 'ALL_PARTNERS',
  gap_pct             NUMERIC(8,3) NOT NULL,
  severity             TEXT NOT NULL CHECK (severity IN ('quiet','flag','warning','untrustworthy')),
  direction_flip_yoy     BOOLEAN NOT NULL DEFAULT false,
  PRIMARY KEY (hs6, flow, year, check_name, partner_country_code)
);

-- [PM-1 fix: MAJOR "mid-year duty-rate changes have no resolution rule"]
-- Computed at MONTHLY granularity, not yearly — a Budget-day rate change
-- is then visible exactly at the month it took effect, never averaged
-- away or ambiguous about which rate applied. The report's headline
-- "landed cost" figure for a year is explicitly the MOST RECENT month's
-- row within that year, labeled "as of <month>" in the facts JSON (§14) —
-- never a yearly average of a duty rate that only makes sense as a
-- point-in-time figure.
--
-- [2026-08-23 revision: evidence-first duty data, §4a] Every duty-amount
-- column is nullable and stays NULL when its component wasn't VERIFIED —
-- `is_complete`/`excluded_components` say why, rather than the absence
-- being ambiguous between "zero" and "unknown" (the exact D1/D2 rule this
-- whole plan already applies to trade data, now applied here too).
-- `landed_cost_inr_paise_per_kg` is populated only when `is_complete`;
-- `partial_landed_cost_inr_paise_per_kg` is the always-computable,
-- explicitly-partial figure from whichever components are VERIFIED.
CREATE TABLE analytics_landed_cost (
  hs8              TEXT NOT NULL,
  month            DATE NOT NULL,           -- first-of-month
  cif_inr_paise_per_kg BIGINT NOT NULL,
  bcd_inr_paise_per_kg BIGINT,
  aidc_inr_paise_per_kg BIGINT,
  sws_inr_paise_per_kg BIGINT,
  igst_inr_paise_per_kg BIGINT,
  is_complete            BOOLEAN NOT NULL,
  excluded_components      TEXT[] NOT NULL DEFAULT '{}',  -- e.g. {'IGST'} when NOT_VERIFIED/CONFLICTING/EXPIRED
  landed_cost_inr_paise_per_kg BIGINT,          -- NULL unless is_complete
  partial_landed_cost_inr_paise_per_kg BIGINT,  -- always populated from whatever IS verified
  -- [PM-1 fix: MINOR "Agmarknet per-quintal never converted to per-kg"]
  -- Converted once, at the Agmarknet normalizer boundary (D7's "convert
  -- only what needs converting" — do it once, at ingestion, never
  -- mid-calculation): raw_agmarknet_prices.modal_price_inr_paise_per_qtl / 100
  -- becomes normalized_trade_flows-equivalent per-kg value before it ever
  -- reaches this table. landed_cost.py only ever reads already-per-kg values.
  domestic_price_inr_paise_per_kg BIGINT,     -- NULL if Agmarknet coverage too thin (see §1 gap)
  margin_pct          NUMERIC(8,3),           -- NULL unless is_complete AND domestic price known
  domestic_price_confidence TEXT NOT NULL CHECK (domestic_price_confidence IN ('good','limited','unavailable')),
  PRIMARY KEY (hs8, month)
);
-- Full per-component evidence (status, citation, verified_date, notes) for
-- a given (hs8, month) is a join against ref_duty_components on whichever
-- row's [effective_from, effective_to) window covers that month — not
-- duplicated into this table, which only holds the already-computed
-- amounts. The facts JSON (§14) assembles both together.

CREATE TABLE analytics_coverage_summary (                  -- D11
  hs6              TEXT NOT NULL,
  flow             TEXT NOT NULL,
  window_start       DATE NOT NULL,
  window_end          DATE NOT NULL,
  expected_cells        INT NOT NULL,
  present_cells           INT NOT NULL,
  not_yet_published_cells INT NOT NULL,
  suppressed_cells         INT NOT NULL,
  fetch_failed_cells         INT NOT NULL,
  gate_passed                  BOOLEAN NOT NULL,
  degraded                      BOOLEAN NOT NULL,  -- D3: fetch_failed > 5% of expected
  PRIMARY KEY (hs6, flow, window_start, window_end)
);
```

## 4a. Evidence-first duty data (2026-08-23 revision, user-directed)

**Primary sources only**: ICEGATE's Trade Guide on Imports / Customs Duty Calculator, and the CBIC Tax
Information Portal (`taxinformation.cbic.gov.in`). Both verified to be real, official government sites, but
**neither exposes a public API** — both are human-facing search/calculator tools (confirmed: ICEGATE's own
description is "search by text or Customs Tariff Head... shows all applicable duties per Tariff"; CBIC's
portal is a notification/circular search interface). A live fetch attempt against both also hit TLS
certificate-verification failures from this environment — a concrete signal that even a best-effort scraper
would be fragile, not just unofficial. **Decision: no scraper for v1.** Duty verification is a deliberate
manual-curation workflow, not an ingestion job — consistent with this plan's own non-goals section already
saying duty rates are "a maintained reference table... an operator updates it, same pattern as the
taxonomy CSV," now made evidence-strict rather than a flat trusted CSV.

**`DutySource` adapter** (`app/pipeline/duty_source.py`), the seam that makes the verification mechanism
swappable later without touching `landed_cost.py`:

```python
class ConflictCandidate(BaseModel):
    value_pct: Decimal
    source_authority: str
    source_reference: str
    source_url: str | None

class DutyComponentEvidence(BaseModel):
    component: Literal["BCD", "AIDC", "SWS", "IGST"]
    verification_status: Literal["VERIFIED", "NOT_VERIFIED", "CONFLICTING", "EXPIRED"]
    value_pct: Decimal | None       # None unless VERIFIED or EXPIRED — enforced by validator, not just convention
    source_authority: str
    source_reference: str
    source_url: str | None
    verified_date: date
    notes: str | None
    conflicting_candidates: list[ConflictCandidate] | None  # populated only when CONFLICTING

class DutyEvidence(BaseModel):
    hs8: str
    as_of: date
    components: dict[Literal["BCD", "AIDC", "SWS", "IGST"], DutyComponentEvidence]

class DutySource(Protocol):
    async def get_duty_evidence(self, hs8: str, *, as_of: date) -> DutyEvidence: ...
```

`ManualDutySource` (v1, only implementation): reads `ref_duty_components`/`ref_duty_component_conflicts`
(§4) for the row(s) covering `as_of`. **Populating those tables is a human task**: a curator looks up the
current rate on ICEGATE/CBIC, records the citation via a small CLI (`scripts/record_duty_rate.py`, to be
built alongside this) that writes one `ref_duty_components` row and — on entering a new `VERIFIED` row for
a component that already has a current one — atomically sets the old row's `effective_to`/flips it to
`EXPIRED` in the same transaction. A component with no row at all for a given `hs8` returns
`verification_status='NOT_VERIFIED'` by construction (the query's default, not a stored row) — there is
never a code path that invents a 0% or omits the component silently.

**`landed_cost.py`'s contract changes from "compute a number" to "compute an evidence-aware result":**

```python
class LandedCostResult(BaseModel):
    is_complete: bool                              # False if any component isn't VERIFIED
    landed_cost_inr_paise_per_kg: int | None        # None when is_complete=False — never a partial total
    partial_landed_cost_inr_paise_per_kg: int | None  # computed from only the VERIFIED components, clearly labeled
    excluded_components: list[str]                  # which components were excluded and why (their status)
    components: dict[str, DutyComponentEvidence]     # every component's full evidence, always
```

`compute_landed_cost(cif_per_kg, evidence: DutyEvidence) -> LandedCostResult`: **never** substitutes 0% or
any guessed value for a `NOT_VERIFIED`/`CONFLICTING`/`EXPIRED` component. If every component is `VERIFIED`,
`is_complete=True` and `landed_cost_inr_paise_per_kg` is populated as before (§11's formula). Otherwise
`is_complete=False`, `landed_cost_inr_paise_per_kg=None`, and `partial_landed_cost_inr_paise_per_kg` holds a
clearly-separate, explicitly-partial figure computed only from the components that *are* `VERIFIED` (useful
context, never presented as "the" landed cost) — this is the direct implementation of the user's rule "you
may show a clearly labelled partial calculation, but it must explicitly exclude the unverified
component(s)."

**Report/facts JSON** (§14, updated): `landed_cost` gains `is_complete`, `excluded_components`, and a full
`components` breakdown — each with its own `verification_status` and citation shown next to the number,
never a bare figure. The narrative (`report/narrative.py`) is instructed never to state a landed-cost total
when `is_complete=False`, only the partial figure with its explicit caveat — enforced the same way as
D4's number-grounding validator (a stated total that doesn't match a `VERIFIED`-backed computation is
rejected).

## 5. Status enum — where each value is set

| Value | Set by | Condition |
|---|---|---|
| `OK` | normalizer (all sources) | real value present, unit/period recognized |
| `ZERO` | normalizer | source explicitly reports a numeric `0`, distinct from an absent row |
| `NOT_REPORTED` | normalizer | period exists for the series but this partner/period combination is absent from the source response |
| `SUPPRESSED` | normalizer (DGCIS/Comtrade) | source marks the cell confidential/withheld (explicit source flag, never inferred) |
| `NOT_YET_PUBLISHED` | normalizer | `period_month` is after the source's last-updated watermark; `status_detail` carries the expected publication window from the source's own cadence |
| `PROVISIONAL` | normalizer | period is within the source's own "subject to revision" window (DGCIS: current + prior 2 months; Comtrade: `isReported`/preliminary flag from payload) |
| `QTY_MISSING` | normalizer | `value` present, `quantity` null |
| `UNIT_MISMATCH` | `unit_consistency.py` (D10) at HS8→HS6 rollup time | sibling HS8 lines report different units |
| `CODE_RETIRED` | `dgcis.py` normalizer | HS8 code absent from the current nomenclature vintage but present in raw history |
| `FETCH_FAILED` | ingestion job's own error handler, **before** any normalization | any exception surfaces here; also written to `dead_letter_ingestion` |

`ZERO` vs missing is enforced structurally: the normalizer only ever writes `ZERO` from an explicit
numeric `0` in the source payload; every "row absent" path writes `NOT_REPORTED`/`FETCH_FAILED`/etc, never
`ZERO` by default. A dedicated test (§10) constructs both cases from the same fixture family and asserts
they never cross.

## 6. FX design (fully resolved per §1's live verification)

```
fx/client.py    — get_rate(date: date) -> Decimal, calls Frankfurter's /v2/rate/USD/INR?date=...
fx/cache.py     — get_or_fetch(date) -> (Decimal, is_stale: bool, actual_date: date)
                   - today  -> cache key fx:USD:INR:<today>, TTL to next IST midnight
                   - past   -> cache key fx:USD:INR:<date>, no TTL (immutable)
                   - on Frankfurter failure -> return most recent cached entry (scan back up to 7 days),
                     is_stale=True, actual_date = that entry's own date
```
One call per (date) per process lifetime, enforced by the cache check happening *before* any client call —
tested by asserting exactly one `httpx` call across two `get_or_fetch` calls for the same date (D8's
"second call for a cached date is a MAJOR").

Three-way decomposition (`fx/decomposition.py`), for the D8 report requirement:
```
value_inr(t) = qty(t) × price_native(t) × fx(t)      [fx(t) = 1 for DGCIS's native INR rows]
Δvalue% ≈ Δqty% + Δprice_native% + Δfx%               (first-order log-decomposition)
```
Computed only from `normalized_trade_flows` rows already carrying `fx_rate_used`/`fx_rate_date` (DGCIS
rows have `fx_rate_used = NULL`, contribute `0` to the FX term by construction — never round-tripped
through USD, per D8's explicit `BLOCKER`).

## 7. Ingestion jobs

| Source | Cadence | Idempotency | Failure handling |
|---|---|---|---|
| DGCIS | Monthly (scheduled) | Upsert on `(fiscal_year, calendar_month, hs8, flow, partner_country)` — re-running a month overwrites `raw_dgcis_monthly` for that key, never appends duplicates | Per-request failures → `dead_letter_ingestion`, job continues to next country/month rather than aborting the batch |
| Comtrade mirror | Nightly, tracked HS6 codes only | Upsert on `(period, reporter_code, partner_code, flow_code, cmd_code)` | D6 retry schedule + circuit breaker; exhausted retries → dead-letter, `FETCH_FAILED` cell |
| BACI | Annual (new vintage detection — check CEPII page for a new vintage id before downloading) | Upsert on `(vintage, year, exporter_code, importer_code, hs6)`; a vintage is immutable once loaded | Download failure → dead-letter, prior vintage stays authoritative until a retry succeeds |
| Duty table | Manual, curator-driven (`scripts/record_duty_rate.py`, §4a) — not an ingestion job at all | N/A — one row inserted per curator action, closing out the previous current row atomically | N/A — CLI validation rejects malformed input before touching the database |
| Agmarknet | Daily | Upsert on `(price_date, commodity, market)` | Same dead-letter pattern; **blocked on a real API key (§1)** |

**DGCIS's real request volume, per §1's live findings**: no report returns a commodity×country cross-tab
in one call — the annual job loops over India's ~250 real partner-country codes per tracked HS8 code, via
`commodityx_countries_wise_import`/`_export` (real, verified fields:
`searchTerm`/`ContEidbi`/`ContEidbyi`/`ReportEidbi`). Each of those ~250 requests returns a full 5-year
annual series in one response (verified live), so the annual refresh is **~250 requests per tracked HS8
code, not per period** — materially more tractable than first feared, though still real, non-trivial
volume for a portal with no documented rate limit, needing its own rate-limit/backoff design (a
token-bucket limiter + the same fixed-schedule-retry idea as D6, tuned empirically against the live site's
actual tolerance, not assumed free). The monthly refresh (D15's current-year section) still needs its own
batching-behavior verification before assuming the same generous multi-period-per-call pattern.

Every job proves idempotency with a "run twice, assert identical row count and content" test (§10).

## 8. Comtrade request shape / retry / limiter / breaker (D5, D6)

**[PM-1 fix: BLOCKER "check A cannot be computed from the query design"]** One query shape is not enough —
check A (DGCIS vs India's *own* Comtrade submission) and check B (DGCIS vs *partner's* Comtrade submission)
need India in different roles. Two query shapes, both still "one call, not fifty" (D5's actual requirement
is *no per-country loop*, not *exactly one call ever*):

```
Query 1 — India as REPORTER (feeds check A: India's own submission):
GET {base}/data/v1/get/C/A/HS
  ?reporterCode=699
  &partnerCode=          (omitted = all partners — verified live, §1)
  &period=2021,2022,2023,2024,2025   (comma-joined — verified live, §1)
  &cmdCode=<tracked hs6 list, comma-joined>   (verified live, §1: cmdCode itself also batches)
  &flowCode=M,X            (verified live, §1)

Query 2 — India as PARTNER (feeds check B: each partner's own submission about trade with India):
GET {base}/data/v1/get/C/A/HS
  ?reporterCode=          (omitted = all reporters — verified live, §1)
  &partnerCode=699
  &period=2021,2022,2023,2024,2025
  &cmdCode=<tracked hs6 list, comma-joined>
  &flowCode=M,X
```
Both write into the same `raw_comtrade_records` table (§4) — which query a row came from is always
recoverable from whether `reporter_code` or `partner_code` equals `'699'`, no new column needed. Ranking
(which partners matter) happens entirely in our own code from the returned rows — never a per-partner
request. Retry: fixed schedule `[30, 60, 120, 300]` seconds ±20% jitter, `Retry-After` header overrides the
schedule entry when present, token-bucket rate limiter sized to Comtrade's documented per-minute quota (to
be confirmed against the real key's tier in Step 3), circuit breaker opens after 3 consecutive 429s and
pauses the worker 15 minutes — shared across both query shapes, since they hit the same rate limit. All
retries run inside the background job — the mirror job is never invoked from a request path (this also
satisfies D13).

## 9. Coverage gate (D11)

```
gate(hs6, flow, window) -> passed: bool, reason: str
  expected_cells = months_in_window × tracked_partners_in_scope
  qty_missing_pct = count(status == QTY_MISSING) / expected_cells
  if qty_missing_pct > 0.30: unit_value metric is NOT emitted for this (hs6, flow, window);
                             analytics_unit_value_series.coverage_gate_passed = false, no row's
                             unit_value populated — refuse, don't approximate.
```
Runs against the **selected** window (D14) — a `docs/PLAN.md`-frozen 5y/top-10 default recomputes nothing
extra; an 8-year request recomputes the gate over the wider window and may fail where the 5-year one
passed, since older DGCIS months carry more HS-revision churn (§1's April-2026 finding makes this concrete,
not theoretical).

## 10. Mismatch checks (D9)

**[PM-1 fix: MAJOR "no described join path from normalized_trade_flows to analytics_mismatch_checks"]**
Owned by `report/mismatch.py:compute_checks(hs6, flow, year)`, run as part of the same ingestion pass that
writes `analytics_partner_rankings` (so mismatch checks are always precomputed, never request-time). Join
is always `normalized_trade_flows` filtered to `hs6, flow, period_month within year`, grouped by
`partner_country_code` (via `ref_country_crosswalk`, §4 — this is exactly the join finding #2 flagged as
the highest-risk spot, so it's the *only* place this join happens, not re-implemented per check):
```
check_A(year)          = groupby(source) -> compare 'dgcis' total vs 'comtrade' rows WHERE reporter=India (Query 1, §8)
check_B(year, partner) = compare 'dgcis' row for that partner vs 'comtrade' row WHERE reporter=partner (Query 2, §8)
check_C(year)           = compare 'dgcis' total vs 'baci' total (FOB, already CIF/FOB-adjusted by CEPII)

check_A(year) = |dgcis_total - comtrade_india_reported_total| / dgcis_total
check_B(year, partner) = |dgcis_import - partner_comtrade_export| / dgcis_import   # expect 5-12%, quiet
check_C(year) = (baci_fob_total - dgcis_cif_total) / dgcis_cif_total               # expect BACI < DGCIS
```
A partner with `partner_country_code = 'UNMAPPED'` (§4) is excluded from check B individually and folded
into check A/C's aggregate totals only — an unmapped country cannot be blamed for a specific partner-level
gap it can't be identified for.

severity:
  gap < 15%             -> quiet note
  15% <= gap < 40%       -> flag
  gap >= 40%              -> prominent warning
  sign flips year-on-year  -> data quality warning, series marked untrustworthy (independent of gap size)
```
Boundary tests at 14.9/15.1/39.9/40.1% (§ testing) prove the bands are `<`/`>=`, not off-by-one. Check B's
5–12% band is asserted to render as `quiet`, explicitly — a regression test guards against someone "fixing"
it into a flag later.

## 11. Metrics and formulas

- **Unit value** (₹/kg) = `value_inr_paise / quantity_kg`, only when `coverage_gate_passed` (§9).
- **CAGR** = `(end/start)^(1/years) - 1` over `OK`/`PROVISIONAL`-excluded annual totals; a provisional year
  is never the `end` point of a CAGR without an explicit flag on the output.
- **HHI** (partner concentration) = `Σ(share_i²)` over the full ranked list (§ D14 precompute), computed
  once per `(hs6, flow, year)`, independent of the request's chosen top-N.
- **Landed cost/kg** = `cif_inr_paise_per_kg × (1 + bcd_pct + aidc_pct + sws_pct) × (1 + igst_pct)` (duty on
  duty-inclusive base, matching how BCD/AIDC/SWS compound before IGST is applied — verify the exact
  compounding order against a real CBIC worked example in Step 3, flagged, not assumed here). **Only
  computed when every one of BCD/AIDC/SWS/IGST is `VERIFIED`** (§4a) — otherwise `landed_cost_inr_paise_per_kg`
  stays `NULL` and `partial_landed_cost_inr_paise_per_kg` is computed from whichever components are
  `VERIFIED`, explicitly labeled partial with `excluded_components` naming the rest. Computed monthly
  (§4's revised `analytics_landed_cost`); the report's headline figure is the most recent month within the
  selected window, labeled "as of `<month>`" rather than a yearly average.
- **Margin** = `(domestic_price_per_kg - landed_cost_per_kg) / domestic_price_per_kg`, computed only when
  `is_complete` **and** the domestic price is known — `domestic_price_confidence` copied from the Agmarknet
  coverage check (thin mandi coverage → `limited`, never silently `good`). A margin is never computed
  against a partial landed-cost figure (that would silently understate real cost).
- **FX decomposition**: §6.

## 12. D14 — parameters end to end

`years: int` (1-8, default 5), `top_n: int` (3-25, default 10) flow from the API request body →
`report/service.py` → `analytics_partner_rankings` slice (`WHERE rank <= top_n`) and `analytics_*` window
filters (`WHERE year >= current_year - years`). No literal `5`/`10` below the route handler's own default
values.

**[PM-2 fix: MAJOR "'all other partners' not restated for the nullable-rank schema"]** "All other partners"
is built from the **full partner-universe row set** for `(hs6, flow, year)`, split into three groups, not
one sum:
1. `rank > top_n` (real, comparable values beyond the cutoff) — summed into the value.
2. `rank IS NULL` with a status that has no value at all (`NOT_REPORTED`, `SUPPRESSED`, `NOT_YET_PUBLISHED`,
   `FETCH_FAILED`) — contribute nothing to the sum (there's nothing to add) but their **status** still
   counts toward the aggregate row's own status.
3. `rank <= top_n` — excluded entirely (already shown individually).

`all_other_partners.value_inr_paise` = Σ of group 1 only. `all_other_partners.status` = `OK` only if every
constituent in groups 1 **and** 2 is `OK`/`ZERO`; otherwise the worst status present across both groups
(so a `FETCH_FAILED` partner hiding outside the top-N still surfaces as degraded status on the aggregate
row, not silently absorbed into a clean-looking sum). Totals still reconcile: top-N shown + "all other
partners" value + (group 2's implicit zero contribution, which is honest — there is genuinely no value to
add for a `NOT_REPORTED` partner) always equals the sum of every `OK`/`ZERO` row in the full universe.

## 13. D15 — current-year month-wise section

`analytics_monthly_current_year` is written by the DGCIS ingestion job as part of its normal monthly run
(no separate job) — one row per month of the current calendar year, always, even before that month's data
exists (`status = NOT_YET_PUBLISHED`, `status_detail` = DGCIS's own stated next-update cadence). Rendered
as its own report section, never merged into the annual series (different confidence, per the prompt).
`data_as_of` = the ingestion job's own `scraped_at` for the most recent row, surfaced prominently.

## 14. Facts JSON schema (frozen LLM contract, D4)

```json
{
  "hs6": "120791",
  "product_label": "Poppy seeds",
  "window": {"years": 5, "start_year": 2021, "end_year": 2025},
  "top_n": 10,
  "annual_series": [
    {"year": 2021, "flow": "import", "total_inr_paise": 0, "status": "OK", "partners": [
      {"rank": 1, "country": "Turkey", "value_inr_paise": 0, "status": "OK"}
    ], "all_other_partners": {"value_inr_paise": 0, "status": "OK"}}
  ],
  "month_wise_current_year": [
    {"month": "2026-01", "value_inr_paise": 0, "status": "OK", "mom_change_pct": null, "yoy_change_pct": null}
  ],
  "unit_value_trend": [
    {"year": 2021, "inr_paise_per_kg": 0, "delta_qty_pct": 0, "delta_price_pct": 0, "delta_fx_pct": 0}
  ],
  "hhi_by_year": [{"year": 2021, "hhi": 0.0}],
  "landed_cost": {
    "as_of_month": "2025-11",
    "is_complete": false,
    "inr_paise_per_kg": null,
    "partial_inr_paise_per_kg": 0,
    "excluded_components": ["IGST"],
    "components": {
      "BCD": {"verification_status": "VERIFIED", "value_pct": 20.0,
        "source_authority": "ICEGATE Trade Guide on Imports", "source_reference": "<citation>",
        "verified_date": "2026-08-23"},
      "AIDC": {"verification_status": "VERIFIED", "value_pct": 0.0,
        "source_authority": "ICEGATE Trade Guide on Imports", "source_reference": "<citation>",
        "verified_date": "2026-08-23"},
      "SWS": {"verification_status": "NOT_VERIFIED", "value_pct": null,
        "notes": "Not verified from an authoritative official source."},
      "IGST": {"verification_status": "CONFLICTING", "value_pct": null,
        "notes": "Two official sources disagree; excluded from any complete landed-cost figure pending manual review.",
        "conflicting_candidates": [
          {"value_pct": 5.0, "source_authority": "CBIC Tax Information Portal", "source_reference": "<citation A>"},
          {"value_pct": 12.0, "source_authority": "CBIC Tax Information Portal", "source_reference": "<citation B>"}
        ]}
    },
    "domestic_price_inr_paise_per_kg": null, "margin_pct": null, "domestic_price_confidence": "limited"
  },
  "mismatch_checks": [
    {"check": "B_dgcis_vs_partner_comtrade", "year": 2025, "partner": "Turkey", "gap_pct": 9.1, "severity": "quiet"}
  ],
  "regulatory_note": "CBN contract registration required; imports permitted only from a restricted origin list.",
  "regulatory_note_missing_warning": false,
  "coverage": {"expected_cells": 0, "present_cells": 0, "not_yet_published": 0, "suppressed": 0,
                "fetch_failed": 0, "degraded": false},
  "hs8_split_note": "12079100 is the only ITC-HS8 line beneath 120791 as of this vintage (from ref_hs6_hs8_crosswalk) — DGCIS's value here is frequency, not added granularity."
}
```
**[PM-1 fix: BLOCKER "check B has no partner dimension"]**: every `mismatch_checks` entry now carries
`"partner"` (§4's DDL fix); A/C entries carry `"partner": "ALL_PARTNERS"`.

**[PM-1 fix: MAJOR "D12 enforcement gate"]**: `regulatory_note_missing_warning` is `true` when
`ref_regulatory_notes` has no row for this hs6 **and** the latest `hhi_by_year` value exceeds a concentration
threshold (top-1 partner share > 60%) — a genuine, checkable proxy for "this market might be regulated, not
just commercially concentrated." **[PM-2 fix: MINOR "60% threshold is invented, not verified"]** Unlike
D9's 15%/40% mismatch bands and D11's 30% coverage floor (both given directly by the master prompt), this
60% figure is my own reasoned starting point, not empirically validated against real concentration data
across tracked HS6 codes — flagged here the same way §1 flags other unverified numbers, and it's the first
thing to revisit if it fires too often (every genuinely concentrated-but-unregulated commodity) or too
rarely (a regulated market that just misses the threshold) once real data exists. `report/narrative.py`'s
system prompt hard-rules: when this flag is true,
the model must describe partner concentration neutrally ("origin is concentrated; reason not on file") and
is explicitly forbidden from offering a commercial-preference explanation — this is the structural guard
D12 was missing, not just a maintained-data-exists-somewhere hope.

Every numeral the LLM is allowed to state must trace to a field in this document. `report/narrative.py`'s
post-validator extracts every number from the model's prose and asserts membership against a flattened set
of every numeric leaf in this JSON (mirrors the existing `app.guardrails.check_numbers_grounded` pattern
already used by the shipped `/messages` flow — reused, not reinvented) — reject → regenerate once → template
fallback.

## 15. Indexes (each justified by a named query)

```sql
CREATE INDEX ix_ntf_hs6_flow_period ON normalized_trade_flows (hs6, flow, period_month);
  -- Q: "all rows for hs6=120791, flow=import, across the selected window" — every report request.
CREATE INDEX ix_ntf_hs8_flow_period ON normalized_trade_flows (hs8, flow, period_month);
  -- Q: unit_consistency.py's HS8-sibling lookup before rolling up to HS6 (D10).
CREATE INDEX ix_apr_hs6_flow_year_rank ON analytics_partner_rankings (hs6, flow, year, rank);
  -- Already the PK — listed for clarity; Q: "top N partners for (hs6,flow,year)".
CREATE INDEX ix_dead_letter_source_resolved ON dead_letter_ingestion (source, resolved);
  -- Q: ops dashboard "unresolved failures by source" (D3 alerting).
```
No index proposed without a named query above it — anything else the Architect finds unjustified in Step 4
gets removed.

## 16. Test plan (mapped to the Testing standards)

- `tests/unit/pipeline/test_dgcis_parser.py` — golden HTML fixtures (captured once real scrape mechanics
  are confirmed in Step 3), one fixture per status enum value producible by DGCIS.
- `tests/unit/pipeline/test_status_enum_coverage.py` — parametrized over all 10 enum values, asserting
  each is producible and each renders distinctly (D1's "every enum value has a test").
- `tests/unit/warehouse/test_zero_vs_missing.py` — the dedicated D2 test: same fixture family, one branch
  produces `ZERO`, one produces `NOT_REPORTED`, assert they never collapse through storage → aggregation →
  facts JSON.
- `tests/unit/fx/test_cache.py` — exactly one outbound call per date (mock `httpx`, assert call count),
  historical-date correctness, Frankfurter-unreachable → `FX_STALE` fallback with the fallback's real date,
  no-float-drift money assertions.
- `tests/unit/pipeline/test_comtrade_retry.py` — mocked 429 sequence, assert 30/60/120/300 schedule ± 20%
  jitter bounds, `Retry-After` override, circuit breaker trips after 3 consecutive 429s and pauses 15 min.
- `tests/unit/report/test_mismatch_bands.py` — boundary values 14.9/15.1/39.9/40.1%, direction-flip case.
- `tests/unit/report/test_coverage_gate.py` — 29%/30%/31% `QTY_MISSING` boundary, refusal not approximation.
- `tests/unit/report/test_facts_validator.py` — the D4 test: prose containing a number absent from the
  facts JSON must be rejected.
- `tests/unit/pipeline/test_landed_cost_evidence.py` — §4a's core regression tests: all-VERIFIED produces
  `is_complete=True` with a real total; any single `NOT_VERIFIED`/`CONFLICTING`/`EXPIRED` component forces
  `is_complete=False`, `landed_cost_inr_paise_per_kg=None`, and a `partial_` figure computed from only the
  verified components; a missing component is never silently treated as 0% (assert the computed partial
  total differs from what a naive "missing=0" calculation would produce); `CONFLICTING` never
  auto-selects a candidate value.
- `tests/integration/pipeline/test_idempotency.py` — real Postgres (docker/testcontainers), run each
  ingestion job twice against the same fixture, assert identical row count and content.
- `tests/integration/report/test_parameter_boundaries.py` — years ∈ {1,8,9}, top_n ∈ {3,25,26}, clamp
  behavior when the window exceeds available data.
- `tests/integration/report/test_all_other_partners_reconciles.py` — sum of top-N + "all other partners" ==
  sum of the full ranked list, for a synthetic partner set larger than top-N.
- `tests/unit/pipeline/test_country_crosswalk.py` — an unmapped DGCIS country name writes
  `partner_country_code='UNMAPPED'` (never dropped), routes into "all other partners," and produces a
  `dead_letter_ingestion` row — never a silent misjoin.
- `tests/integration/report/test_partner_universe_persists.py` — the direct regression test for PM-1's
  headline finding: a partner present in year N and absent in year N+1 must appear in
  `analytics_partner_rankings` for year N+1 with `rank=NULL` and a real status (`NOT_REPORTED`), never
  simply missing from the table; a `FETCH_FAILED` case for the same partner/year must be distinguishable
  from the `NOT_REPORTED` case by status alone.
- `tests/unit/report/test_check_a_uses_india_as_reporter.py` — asserts `compute_checks`'s check-A query
  reads `raw_comtrade_records` rows where `reporter_code='699'` (Query 1, §8), not `partner_code='699'`
  rows — the direct regression test for PM-1's check-A finding.
- `tests/unit/report/test_regulatory_note_gate.py` — high-HHI + no `ref_regulatory_notes` row ->
  `regulatory_note_missing_warning=true` in the facts JSON, and a fixed narrative prompt fixture asserts the
  model is instructed not to offer a commercial-preference explanation when the flag is true.
- Unit tests never touch the network (existing repo convention, `MockEmbeddingsClient`/`MockLLM`-style
  fakes extended here for `httpx`/Redis where needed).

## 17. Build sequence (dependency order)

1. `app/warehouse/schema.py` + Alembic migration (empty → full schema) — nothing else can start without it.
2. `app/fx/` (client + cache + decomposition) — fully specified already (§1, §6), no external unknowns left.
3. `app/pipeline/duty_source.py` (§4a: `DutySource` Protocol, `ManualDutySource`, `ref_duty_components`/
   `ref_duty_component_conflicts` access) + `scripts/record_duty_rate.py` (the curator-facing entry-point
   CLI). **No rate is entered until the user supplies a real ICEGATE/CBIC citation for HS 120791** — this
   step ships the mechanism, not fabricated data; the table starts empty (every component genuinely
   `NOT_VERIFIED` by construction) until real evidence is recorded.
4. `app/pipeline/dgcis.py` — **first**, real scrape-mechanics verification against the live form (§1), then
   parser + normalizer + status-enum coverage tests.
5. `app/pipeline/comtrade_mirror.py` — after the D5 live-batching verification (§1).
6. `app/pipeline/baci.py`.
7. `app/pipeline/agmarknet.py` — blocked on the credential gap in §1; build the parser against a
   clearly-labeled provisional fixture in the meantime, wire the real call once a key exists.
8. `app/warehouse/coverage_gate.py`, `app/report/unit_consistency.py`, `app/report/mismatch.py`.
9. `app/report/landed_cost.py`, `app/report/facts.py`.
10. `app/report/narrative.py` (reuses the existing number-grounding guardrail pattern).
11. `routes/trade_report.py` + frontend parameter UI (D14) + month-wise view (D15).
12. Full canonical-scenario dry run against poppy seeds (120791).

## 18. D1–D13 checklist

| # | Addressed in |
|---|---|
| D1 | §5 status enum; every storage/API/UI layer renders `status`, never a raw null/dash |
| D2 | §5, dedicated test in §16 |
| D3 | `dead_letter_ingestion` (§4), `analytics_coverage_summary.degraded` (§4, §9) |
| D4 | §14's facts JSON + `report/narrative.py`'s validator, reusing the existing guardrail pattern |
| D5 | §8, with the honest unverified-batching flag from §1 |
| D6 | §8 |
| D7 | Every `normalized_trade_flows` column (§4) |
| D8 | §6, with the verified Frankfurter-behavior deviation recorded in §1 |
| D9 | §10, with the two-query Comtrade design (§8) and the partner dimension (§4) fixed in iteration 2 |
| D10 | §5 (`UNIT_MISMATCH`), §9 gate ordering (unit check before rollup) |
| D11 | §9 |
| D12 | `ref_regulatory_notes` (§4) + the `regulatory_note_missing_warning` enforcement gate (§14, added iteration 2) |
| D13 | `routes/trade_report.py` never calls a `pipeline/*` job synchronously; an untracked HS6 returns
        `NOT_TRACKED` + an enqueue option, mirroring the existing repo's "ingestion and query are separate
        planes" absence today (there is currently no ingestion at all in this repo — this plan introduces
        the first case where the distinction matters) |

D14/D15 are addressed in §12/§13 respectively (kept out of this table since they're each their own section
per the prompt's structure, not a single-line concern).

VERDICT: N/A — this file is a plan artifact, not a review artifact; Step 2 renders the verdict on it.
