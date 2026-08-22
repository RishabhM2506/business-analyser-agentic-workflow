# PLAN — India Trade Analysis Pipeline (DGCIS-backbone)

Status: DRAFT — Step 1, iteration 1. Awaiting Step 2 (PM) review.

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

**DGCIS Tradestat — verified, form-based, no API:**
- Confirmed: no documented public API; a form-driven HTML report generator
  (`tradestat.commerce.gov.in`). Report types include commodity-wise (2/4/6/8-digit), country-wise,
  commodity×country cross-tabs, values in USD/INR/quantity.
- **Coverage is stated as Indian *fiscal* years ("2017-2018 to 2025-2026"), not calendar years** — the
  prompt's "Jan 2018 onward" is a simplification. FY2017-18 starts April 2017. The exact first available
  *calendar* month must be confirmed against a real scrape in Step 3, not assumed as Jan 2018.
- **Directly confirms D10/CODE_RETIRED is not hypothetical**: the site itself states "ITC HS Code of the
  Commodity is either dropped or re-allocated and the unit of the commodity may be changed from April
  2026" — i.e. an HS-line/unit revision is already in effect as of this plan's date. The scraper and
  normalizer must handle a mid-series code change from day one, not as a later hardening pass.
- Scrape mechanics (exact form fields, session/cookie handling, pagination, response format — HTML table
  vs. downloadable CSV/XLS) are **not yet verified against the live form flow** and must be a first Step 3
  task, recorded in `BUILD-LOG.md` — not guessed here.

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

**UN Comtrade bulk-batching (D5) — partially unverified:**
- The existing `comtrade_client.py` confirms the base request/response shape but was written for
  single-value params. Comtrade's public docs pages were not fetchable in this environment (404/JS-gated).
  **D5's exact claim — one call carrying all reporters + all periods + both flows — must be proven with a
  real live call in Step 3** (comma-separated `period`, `flowCode=M,X`, omitted/`all` `reporterCode`) before
  the ingestion job is built assuming it works exactly that way. If the live API rejects one of those
  batching dimensions, the fallback is to batch the largest dimension that *does* work and keep the others
  single-valued — recorded in `BUILD-LOG.md`, not silently reverting to D5's forbidden per-country loop
  without a note.

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
    duty_table.py       # loads data/duty-rates.csv into ref_duty_rates
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
    landed_cost.py          # CIF + duty table → landed cost
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
CREATE TABLE ref_duty_rates (
  hs8              TEXT NOT NULL,
  effective_from   DATE NOT NULL,
  effective_to     DATE,                    -- NULL = still current
  bcd_pct          NUMERIC(6,3) NOT NULL,
  aidc_pct         NUMERIC(6,3) NOT NULL DEFAULT 0,
  surcharge_pct    NUMERIC(6,3) NOT NULL DEFAULT 0,
  igst_pct         NUMERIC(6,3) NOT NULL,
  source_note      TEXT NOT NULL,           -- citation, e.g. "Budget 2024 notification no. ..."
  updated_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
  PRIMARY KEY (hs8, effective_from)
);

CREATE TABLE ref_regulatory_notes (         -- D12
  hs6              TEXT PRIMARY KEY,
  note             TEXT NOT NULL,
  updated_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_by       TEXT NOT NULL
);

-- ── Raw layer: immutable, append-only, mirrors source shape ──────────────
CREATE TABLE raw_dgcis_monthly (
  id               BIGSERIAL PRIMARY KEY,
  scraped_at       TIMESTAMPTZ NOT NULL,
  fiscal_year      TEXT NOT NULL,           -- "2024-2025" as DGCIS reports it
  calendar_month   DATE NOT NULL,           -- first-of-month, derived from fiscal_year + month label
  hs8              TEXT NOT NULL,
  flow             TEXT NOT NULL CHECK (flow IN ('import','export')),
  partner_country  TEXT NOT NULL,           -- DGCIS's own country name string, not yet normalized
  value_inr_paise  BIGINT,                  -- NULL if the source cell itself was blank/dash
  quantity         NUMERIC(18,3),
  unit             TEXT,
  raw_payload      JSONB NOT NULL,          -- full scraped row, for replay/debugging
  UNIQUE (fiscal_year, calendar_month, hs8, flow, partner_country)
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
CREATE TABLE analytics_partner_rankings (              -- D14 precompute strategy
  hs6              TEXT NOT NULL,
  flow             TEXT NOT NULL,
  year             INT NOT NULL,
  rank              INT NOT NULL,
  partner_country_code TEXT NOT NULL,
  value_inr_paise       BIGINT NOT NULL,
  status                 cell_status NOT NULL,
  PRIMARY KEY (hs6, flow, year, rank)
  -- Full ranked list per (hs6,flow,year), every partner. Sliced to top-N at
  -- query time (D14) — never precomputed per (years,topN) combination.
);

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

CREATE TABLE analytics_mismatch_checks (                  -- D9
  hs6              TEXT NOT NULL,
  flow             TEXT NOT NULL,
  year             INT NOT NULL,
  check_name        TEXT NOT NULL CHECK (check_name IN ('A_dgcis_vs_comtrade_india','B_dgcis_vs_partner_comtrade','C_dgcis_vs_baci')),
  gap_pct             NUMERIC(8,3) NOT NULL,
  severity             TEXT NOT NULL CHECK (severity IN ('quiet','flag','warning','untrustworthy')),
  direction_flip_yoy     BOOLEAN NOT NULL DEFAULT false,
  PRIMARY KEY (hs6, flow, year, check_name)
);

CREATE TABLE analytics_landed_cost (
  hs8              TEXT NOT NULL,
  year             INT NOT NULL,
  cif_inr_paise_per_kg BIGINT NOT NULL,
  bcd_inr_paise_per_kg BIGINT NOT NULL,
  aidc_inr_paise_per_kg BIGINT NOT NULL,
  surcharge_inr_paise_per_kg BIGINT NOT NULL,
  igst_inr_paise_per_kg BIGINT NOT NULL,
  landed_cost_inr_paise_per_kg BIGINT NOT NULL,
  domestic_price_inr_paise_per_kg BIGINT,     -- NULL if Agmarknet coverage too thin (see §1 gap)
  margin_pct          NUMERIC(8,3),
  domestic_price_confidence TEXT NOT NULL CHECK (domestic_price_confidence IN ('good','limited','unavailable')),
  PRIMARY KEY (hs8, year)
);

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
| DGCIS | Monthly (scheduled) | Upsert on `(fiscal_year, calendar_month, hs8, flow, partner_country)` — re-running a month overwrites `raw_dgcis_monthly` for that key, never appends duplicates | Per-request failures → `dead_letter_ingestion`, job continues to next HS8/month rather than aborting the batch |
| Comtrade mirror | Nightly, tracked HS6 codes only | Upsert on `(period, reporter_code, partner_code, flow_code, cmd_code)` | D6 retry schedule + circuit breaker; exhausted retries → dead-letter, `FETCH_FAILED` cell |
| BACI | Annual (new vintage detection — check CEPII page for a new vintage id before downloading) | Upsert on `(vintage, year, exporter_code, importer_code, hs6)`; a vintage is immutable once loaded | Download failure → dead-letter, prior vintage stays authoritative until a retry succeeds |
| Duty table | On file change (`data/duty-rates.csv`, committed reference, loaded at deploy) | Upsert on `(hs8, effective_from)` | Validation failure (malformed row) fails the load loudly, does not partially apply |
| Agmarknet | Daily | Upsert on `(price_date, commodity, market)` | Same dead-letter pattern; **blocked on a real API key (§1)** |

Every job proves idempotency with a "run twice, assert identical row count and content" test (§10).

## 8. Comtrade request shape / retry / limiter / breaker (D5, D6)

Design (pending the live-batching verification flagged in §1):
```
GET {base}/data/v1/get/C/A/HS
  ?reporterCode=          (omitted = all reporters, to be confirmed live)
  &partnerCode=699        (India, always — this is a mirror of others' trade WITH India)
  &period=2021,2022,2023,2024,2025   (comma-joined, to be confirmed live)
  &cmdCode=<tracked hs6 list, comma-joined>
  &flowCode=M,X            (to be confirmed live)
```
Ranking (which partners matter) happens entirely in our own code from the returned rows — never a
per-partner request. Retry: fixed schedule `[30, 60, 120, 300]` seconds ±20% jitter, `Retry-After` header
overrides the schedule entry when present, token-bucket rate limiter sized to Comtrade's documented
per-minute quota (to be confirmed against the real key's tier in Step 3), circuit breaker opens after 3
consecutive 429s and pauses the worker 15 minutes. All retries run inside the background job — the mirror
job is never invoked from a request path (this also satisfies D13).

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

```
check_A(year) = |dgcis_total - comtrade_india_reported_total| / dgcis_total
check_B(year, partner) = |dgcis_import - partner_comtrade_export| / dgcis_import   # expect 5-12%, quiet
check_C(year) = (baci_fob_total - dgcis_cif_total) / dgcis_cif_total               # expect BACI < DGCIS

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
- **Landed cost/kg** = `cif_inr_paise_per_kg × (1 + bcd_pct + aidc_pct + surcharge_pct) × (1 + igst_pct)`
  (duty on duty-inclusive base, matching how BCD/AIDC/surcharge compound before IGST is applied — verify
  the exact compounding order against a real CBIC worked example in Step 3, flagged, not assumed here).
- **Margin** = `(domestic_price_per_kg - landed_cost_per_kg) / domestic_price_per_kg`, `domestic_price_confidence`
  copied from the Agmarknet coverage check (thin mandi coverage → `limited`, never silently `good`).
- **FX decomposition**: §6.

## 12. D14 — parameters end to end

`years: int` (1-8, default 5), `top_n: int` (3-25, default 10) flow from the API request body →
`report/service.py` → `analytics_partner_rankings` slice (`WHERE rank <= top_n`) and `analytics_*` window
filters (`WHERE year >= current_year - years`). No literal `5`/`10` below the route handler's own default
values. "All other partners" row = `Σ` of every rank beyond `top_n` for that `(hs6, flow, year)`, own
`status` = `OK` if every constituent is `OK`, else the "worst" status present among constituents (never
silently dropped — totals reconcile by construction since it's a sum over the *same* full ranked list the
top-N was sliced from).

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
  "landed_cost": {"year": 2025, "inr_paise_per_kg": 0, "domestic_price_inr_paise_per_kg": null,
                    "margin_pct": null, "domestic_price_confidence": "limited"},
  "mismatch_checks": [{"check": "B_dgcis_vs_partner_comtrade", "year": 2025, "gap_pct": 9.1, "severity": "quiet"}],
  "regulatory_note": "CBN contract registration required; imports permitted only from a restricted origin list.",
  "coverage": {"expected_cells": 0, "present_cells": 0, "not_yet_published": 0, "suppressed": 0,
                "fetch_failed": 0, "degraded": false},
  "hs8_split_note": "12079100 is the only ITC-HS8 line beneath 120791 as of this vintage — DGCIS's value here is frequency, not added granularity."
}
```
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
- `tests/integration/pipeline/test_idempotency.py` — real Postgres (docker/testcontainers), run each
  ingestion job twice against the same fixture, assert identical row count and content.
- `tests/integration/report/test_parameter_boundaries.py` — years ∈ {1,8,9}, top_n ∈ {3,25,26}, clamp
  behavior when the window exceeds available data.
- `tests/integration/report/test_all_other_partners_reconciles.py` — sum of top-N + "all other partners" ==
  sum of the full ranked list, for a synthetic partner set larger than top-N.
- Unit tests never touch the network (existing repo convention, `MockEmbeddingsClient`/`MockLLM`-style
  fakes extended here for `httpx`/Redis where needed).

## 17. Build sequence (dependency order)

1. `app/warehouse/schema.py` + Alembic migration (empty → full schema) — nothing else can start without it.
2. `app/fx/` (client + cache + decomposition) — fully specified already (§1, §6), no external unknowns left.
3. `app/pipeline/duty_table.py` + `data/duty-rates.csv` (a real, sourced starter table — flagged: initial
   rates need a real CBIC citation, not fabricated numbers).
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
| D9 | §10 |
| D10 | §5 (`UNIT_MISMATCH`), §9 gate ordering (unit check before rollup) |
| D11 | §9 |
| D12 | `ref_regulatory_notes` (§4), fed into §14's facts JSON |
| D13 | `routes/trade_report.py` never calls a `pipeline/*` job synchronously; an untracked HS6 returns
        `NOT_TRACKED` + an enqueue option, mirroring the existing repo's "ingestion and query are separate
        planes" absence today (there is currently no ingestion at all in this repo — this plan introduces
        the first case where the distinction matters) |

D14/D15 are addressed in §12/§13 respectively (kept out of this table since they're each their own section
per the prompt's structure, not a single-line concern).

VERDICT: N/A — this file is a plan artifact, not a review artifact; Step 2 renders the verdict on it.
