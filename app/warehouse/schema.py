"""SQLAlchemy Core table definitions for the India trade analysis pipeline
(`docs/PLAN.md` §4). Core `Table` objects, not ORM classes — deliberate:
every ingestion job bulk-upserts (`docs/PLAN.md`'s bulk-upsert/COPY
performance requirement), which Core's `insert().on_conflict_do_update()`
supports directly without an ORM session's per-row identity-map overhead.

Postgres-specific by design (JSONB, native ENUM, `NUMERIC`) — this schema
requires a real Postgres `DATABASE_URL`, unlike the rest of this repo's
existing SQLite-tolerant checkpoint-only usage (`app/graph.py`). Alembic
(`migrations/`) is the only thing that creates/alters these tables; this
module is read by both Alembic (`migrations/env.py`) and every ingestion
job/report module, so schema and code can never drift apart.

Column types match `docs/PLAN.md` §4 exactly:
- Money is always `BIGINT` paise (1 INR = 100 paise), never `float` — see
  the plan's D8 "money is never a float" rule.
- Quantities are `NUMERIC(18,3)`, FX rates `NUMERIC(12,6)`, percentages
  `NUMERIC(8,3)`.
"""

from __future__ import annotations

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    Column,
    Date,
    DateTime,
    ForeignKeyConstraint,
    Index,
    Integer,
    MetaData,
    Numeric,
    PrimaryKeyConstraint,
    Table,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from sqlalchemy.dialects.postgresql import ENUM as PGEnum

metadata = MetaData()

# ── Status enum (docs/PLAN.md §5) ────────────────────────────────────────
# Native Postgres ENUM (`CREATE TYPE cell_status AS ENUM (...)`), created
# once by the initial migration. Every value's producing condition is
# documented in PLAN.md §5, not here — this is the wire type only.
CELL_STATUS_VALUES = (
    "OK",
    "ZERO",
    "NOT_REPORTED",
    "SUPPRESSED",
    "NOT_YET_PUBLISHED",
    "PROVISIONAL",
    "QTY_MISSING",
    "UNIT_MISMATCH",
    "CODE_RETIRED",
    "FETCH_FAILED",
)
cell_status_enum = PGEnum(*CELL_STATUS_VALUES, name="cell_status", metadata=metadata)

# ── Reference tables (maintained, not scraped) ───────────────────────────

# [docs/PLAN.md §4a, 2026-08-23 user-directed "evidence-first" revision]
# Replaces an earlier flat, trusted ref_duty_rates table. One row per
# (hs8, component, effective_from) — each duty component (BCD/AIDC/SWS/
# IGST) carries its own verification status and citation independently.
# value_pct is NULL unless verification_status='VERIFIED' (enforced by the
# check constraint below, not just convention) — this is what makes "NULL/
# unknown must never be interpreted as 0%" true by construction.
DUTY_VERIFICATION_STATUS_VALUES = ("VERIFIED", "NOT_VERIFIED", "CONFLICTING", "EXPIRED")
duty_verification_status_enum = PGEnum(
    *DUTY_VERIFICATION_STATUS_VALUES, name="duty_verification_status", metadata=metadata
)
DUTY_COMPONENTS = ("BCD", "AIDC", "SWS", "IGST")

ref_duty_components = Table(
    "ref_duty_components",
    metadata,
    Column("hs8", Text, nullable=False),
    Column("component", Text, nullable=False),
    Column("effective_from", Date, nullable=False),
    Column("effective_to", Date, nullable=True),
    Column("verification_status", duty_verification_status_enum, nullable=False),
    Column("value_pct", Numeric(6, 3), nullable=True),
    Column("source_authority", Text, nullable=False),
    Column("source_reference", Text, nullable=False),
    Column("source_url", Text, nullable=True),
    Column("verified_date", Date, nullable=False),
    Column("notes", Text, nullable=True),
    CheckConstraint("component IN ('BCD','AIDC','SWS','IGST')", name="ck_rdc_component"),
    # EXPIRED also keeps its value_pct ("preserve it for historical
    # analysis") — only NOT_VERIFIED/CONFLICTING have no trustworthy
    # single number. See the matching validator in
    # app.pipeline.duty_source.DutyComponentEvidence.
    CheckConstraint(
        "(verification_status IN ('VERIFIED','EXPIRED') AND value_pct IS NOT NULL) OR "
        "(verification_status IN ('NOT_VERIFIED','CONFLICTING') AND value_pct IS NULL)",
        name="ck_rdc_value_matches_status",
    ),
    PrimaryKeyConstraint("hs8", "component", "effective_from"),
)

# Populated only for a component whose current row is CONFLICTING — the N
# disagreeing official values found, each with its own citation. Never
# auto-resolved into a single ref_duty_components.value_pct (user's
# explicit rule: "do not automatically choose one value").
ref_duty_component_conflicts = Table(
    "ref_duty_component_conflicts",
    metadata,
    Column("id", BigInteger, primary_key=True, autoincrement=True),
    Column("hs8", Text, nullable=False),
    Column("component", Text, nullable=False),
    Column("effective_from", Date, nullable=False),
    Column("candidate_value_pct", Numeric(6, 3), nullable=False),
    Column("source_authority", Text, nullable=False),
    Column("source_reference", Text, nullable=False),
    Column("source_url", Text, nullable=True),
    ForeignKeyConstraint(
        ["hs8", "component", "effective_from"],
        [
            "ref_duty_components.hs8",
            "ref_duty_components.component",
            "ref_duty_components.effective_from",
        ],
    ),
)

ref_regulatory_notes = Table(
    "ref_regulatory_notes",
    metadata,
    Column("hs6", Text, primary_key=True),
    Column("note", Text, nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False, server_default=text("now()")),
    Column("updated_by", Text, nullable=False),
)

# [docs/PLAN.md §4, PM-1 fix: country-code crosswalk] DGCIS's free-text
# country names -> the UN M49 numeric codes Comtrade/BACI use. An unmapped
# name never blocks ingestion (see app/pipeline normalizers) — it's
# written with partner_country_code='UNMAPPED' and dead-lettered instead.
ref_country_crosswalk = Table(
    "ref_country_crosswalk",
    metadata,
    Column("dgcis_country_name", Text, primary_key=True),
    Column("country_code", Text, nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False, server_default=text("now()")),
)

# [docs/PLAN.md §4, PM-1 fix: HS6->HS8 crosswalk] Derived from DGCIS scrape
# responses, not manually maintained — every dgcis ingestion run upserts
# the distinct (hs6, hs8) pairs it actually observed.
ref_hs6_hs8_crosswalk = Table(
    "ref_hs6_hs8_crosswalk",
    metadata,
    Column("hs6", Text, nullable=False),
    Column("hs8", Text, nullable=False),
    Column("first_seen_at", DateTime(timezone=True), nullable=False),
    Column("effective_to", DateTime(timezone=True), nullable=True),
    PrimaryKeyConstraint("hs6", "hs8"),
)

ref_hs_revision_notes = Table(
    "ref_hs_revision_notes",
    metadata,
    Column("hs6", Text, primary_key=True),
    Column("note", Text, nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False, server_default=text("now()")),
)

# ── Raw layer: immutable, append-only, mirrors source shape ─────────────

# [2026-08-23 revision, found via real Step 3 investigation, docs/PLAN.md
# §1/§7] DGCIS genuinely publishes two different shapes, not one: an
# ANNUAL report broken down by partner country (raw_dgcis_annual, below —
# app/pipeline/dgcis.py's commodityx_countries_wise_import/_export), and a
# MONTHLY report with no partner dimension at all (this table,
# meidb/commoditywise_import — not yet built). partner_country here will
# use the 'ALL_PARTNERS' sentinel already established in
# analytics_mismatch_checks, since that source structurally has none to
# record — not a scraping gap.
raw_dgcis_monthly = Table(
    "raw_dgcis_monthly",
    metadata,
    Column("id", BigInteger, primary_key=True, autoincrement=True),
    Column("scraped_at", DateTime(timezone=True), nullable=False),
    Column("fiscal_year", Text, nullable=False),
    Column("calendar_month", Date, nullable=False),
    Column("hs8", Text, nullable=False),
    Column("flow", Text, nullable=False),
    Column("partner_country", Text, nullable=False),
    Column("value_inr_paise", BigInteger, nullable=True),
    Column("quantity", Numeric(18, 3), nullable=True),
    Column("unit", Text, nullable=True),
    Column("raw_payload", JSONB, nullable=False),
    CheckConstraint("flow IN ('import','export')", name="ck_raw_dgcis_monthly_flow"),
    UniqueConstraint(
        "fiscal_year",
        "calendar_month",
        "hs8",
        "flow",
        "partner_country",
        name="uq_raw_dgcis_monthly",
    ),
)

# [2026-08-23, new] Mirrors commodityx_countries_wise_import/_export's real
# shape exactly (app/pipeline/dgcis.py's DgcisAnnualRecord): one row per
# (fiscal_year, hs8, flow, partner_country) — DGCIS's own fiscal-year
# label stored verbatim (e.g. "2020 - 2021"), not yet reframed to a
# calendar month/year (that happens at the normalized layer, D7). No
# `quantity` column: this report only ever returns value, never quantity
# (verified live — QTY_MISSING is the correct status for every row from
# this source, not an omission).
raw_dgcis_annual = Table(
    "raw_dgcis_annual",
    metadata,
    Column("id", BigInteger, primary_key=True, autoincrement=True),
    Column("scraped_at", DateTime(timezone=True), nullable=False),
    Column("fiscal_year_label", Text, nullable=False),
    Column("hs8", Text, nullable=False),
    Column("flow", Text, nullable=False),
    Column("partner_country", Text, nullable=False),
    Column("description", Text, nullable=True),
    Column("unit", Text, nullable=True),
    Column("value_inr_paise", BigInteger, nullable=True),
    Column("raw_payload", JSONB, nullable=False),
    CheckConstraint("flow IN ('import','export')", name="ck_raw_dgcis_annual_flow"),
    UniqueConstraint(
        "fiscal_year_label", "hs8", "flow", "partner_country", name="uq_raw_dgcis_annual"
    ),
)

raw_comtrade_records = Table(
    "raw_comtrade_records",
    metadata,
    Column("id", BigInteger, primary_key=True, autoincrement=True),
    Column("fetched_at", DateTime(timezone=True), nullable=False),
    Column("period", Integer, nullable=False),
    Column("reporter_code", Text, nullable=False),
    Column("partner_code", Text, nullable=False),
    Column("flow_code", Text, nullable=False),
    Column("cmd_code", Text, nullable=False),
    Column("primary_value_usd", Numeric(18, 2), nullable=True),
    Column("net_weight_kg", Numeric(18, 3), nullable=True),
    Column("is_reported", Boolean, nullable=False),
    Column("raw_payload", JSONB, nullable=False),
    UniqueConstraint(
        "period",
        "reporter_code",
        "partner_code",
        "flow_code",
        "cmd_code",
        name="uq_raw_comtrade_records",
    ),
)

raw_baci_records = Table(
    "raw_baci_records",
    metadata,
    Column("id", BigInteger, primary_key=True, autoincrement=True),
    Column("loaded_at", DateTime(timezone=True), nullable=False),
    Column("vintage", Text, nullable=False),
    Column("hs_revision", Text, nullable=False),
    Column("year", Integer, nullable=False),
    Column("exporter_code", Text, nullable=False),
    Column("importer_code", Text, nullable=False),
    Column("hs6", Text, nullable=False),
    Column("value_fob_usd", Numeric(18, 2), nullable=True),
    Column("quantity_kg", Numeric(18, 3), nullable=True),
    UniqueConstraint(
        "vintage", "year", "exporter_code", "importer_code", "hs6", name="uq_raw_baci_records"
    ),
)

raw_agmarknet_prices = Table(
    "raw_agmarknet_prices",
    metadata,
    Column("id", BigInteger, primary_key=True, autoincrement=True),
    Column("fetched_at", DateTime(timezone=True), nullable=False),
    Column("price_date", Date, nullable=False),
    Column("commodity", Text, nullable=False),
    Column("market", Text, nullable=False),
    Column("state", Text, nullable=False),
    Column("modal_price_inr_paise_per_qtl", BigInteger, nullable=True),
    Column("raw_payload", JSONB, nullable=False),
    UniqueConstraint("price_date", "commodity", "market", name="uq_raw_agmarknet_prices"),
)

# ── Dead letter (docs/PLAN.md D3) ────────────────────────────────────────

dead_letter_ingestion = Table(
    "dead_letter_ingestion",
    metadata,
    Column("id", BigInteger, primary_key=True, autoincrement=True),
    Column("source", Text, nullable=False),
    Column("job_run_id", UUID(as_uuid=True), nullable=False),
    Column("attempted_at", DateTime(timezone=True), nullable=False),
    Column("request_desc", Text, nullable=False),
    Column("error_message", Text, nullable=False),
    Column("attempt_count", Integer, nullable=False),
    Column("resolved", Boolean, nullable=False, server_default=text("false")),
)
Index(
    "ix_dead_letter_source_resolved",
    dead_letter_ingestion.c.source,
    dead_letter_ingestion.c.resolved,
)

# ── Normalized layer ──────────────────────────────────────────────────────

normalized_trade_flows = Table(
    "normalized_trade_flows",
    metadata,
    Column("id", BigInteger, primary_key=True, autoincrement=True),
    Column("source", Text, nullable=False),
    Column("hs6", Text, nullable=False),
    Column("hs8", Text, nullable=True),
    Column("hs_revision", Text, nullable=False),
    Column("flow", Text, nullable=False),
    Column("period_month", Date, nullable=False),
    Column("calendar", Text, nullable=False),
    Column("partner_country_code", Text, nullable=False),
    Column("basis", Text, nullable=False),
    Column("currency", Text, nullable=False),
    Column("universe", Text, nullable=False),
    Column("dataset_version", Text, nullable=False),
    Column("is_provisional", Boolean, nullable=False),
    Column("status", cell_status_enum, nullable=False),
    Column("status_detail", Text, nullable=True),
    Column("value_inr_paise", BigInteger, nullable=True),
    Column("value_original_currency_paise", BigInteger, nullable=True),
    Column("fx_rate_used", Numeric(12, 6), nullable=True),
    Column("fx_rate_date", Date, nullable=True),
    Column("quantity_kg", Numeric(18, 3), nullable=True),
    Column("ingested_at", DateTime(timezone=True), nullable=False, server_default=text("now()")),
    CheckConstraint("flow IN ('import','export')", name="ck_ntf_flow"),
    CheckConstraint("calendar IN ('CY','FY')", name="ck_ntf_calendar"),
    CheckConstraint("basis IN ('CIF','FOB')", name="ck_ntf_basis"),
    CheckConstraint("currency IN ('INR','USD')", name="ck_ntf_currency"),
    UniqueConstraint(
        "source",
        "hs6",
        "hs8",
        "flow",
        "period_month",
        "partner_country_code",
        "dataset_version",
        name="uq_normalized_trade_flows",
    ),
)
Index(
    "ix_ntf_hs6_flow_period",
    normalized_trade_flows.c.hs6,
    normalized_trade_flows.c.flow,
    normalized_trade_flows.c.period_month,
)
Index(
    "ix_ntf_hs8_flow_period",
    normalized_trade_flows.c.hs8,
    normalized_trade_flows.c.flow,
    normalized_trade_flows.c.period_month,
)

# ── Analytics layer: precomputed on ingest, API reads only this ──────────

# [docs/PLAN.md §4, PM-1 fix: partner-disappeared vs pipeline-broke] Keyed
# on partner, not rank, so a partner's status persists every year even
# with no data that year (rank/value NULL in that case) — see the PK.
analytics_partner_rankings = Table(
    "analytics_partner_rankings",
    metadata,
    Column("hs6", Text, nullable=False),
    Column("flow", Text, nullable=False),
    Column("year", Integer, nullable=False),
    Column("partner_country_code", Text, nullable=False),
    Column("rank", Integer, nullable=True),
    Column("value_inr_paise", BigInteger, nullable=True),
    Column("status", cell_status_enum, nullable=False),
    PrimaryKeyConstraint("hs6", "flow", "year", "partner_country_code"),
)
Index(
    "ix_apr_rank_where_present",
    analytics_partner_rankings.c.hs6,
    analytics_partner_rankings.c.flow,
    analytics_partner_rankings.c.year,
    analytics_partner_rankings.c.rank,
    unique=True,
    postgresql_where=analytics_partner_rankings.c.rank.isnot(None),
)

analytics_monthly_current_year = Table(
    "analytics_monthly_current_year",
    metadata,
    Column("hs6", Text, nullable=False),
    Column("flow", Text, nullable=False),
    Column("month", Date, nullable=False),
    Column("value_inr_paise", BigInteger, nullable=True),
    Column("status", cell_status_enum, nullable=False),
    Column("status_detail", Text, nullable=True),
    Column("is_provisional", Boolean, nullable=False),
    Column("mom_change_pct", Numeric(8, 3), nullable=True),
    Column("yoy_same_month_pct", Numeric(8, 3), nullable=True),
    Column("data_as_of", DateTime(timezone=True), nullable=False),
    PrimaryKeyConstraint("hs6", "flow", "month"),
)

analytics_unit_value_series = Table(
    "analytics_unit_value_series",
    metadata,
    Column("hs6", Text, nullable=False),
    Column("flow", Text, nullable=False),
    Column("year", Integer, nullable=False),
    Column("unit_value_inr_paise_per_kg", Numeric(18, 4), nullable=True),
    Column("delta_value_pct", Numeric(8, 3), nullable=True),
    Column("delta_from_qty_pct", Numeric(8, 3), nullable=True),
    Column("delta_from_price_pct", Numeric(8, 3), nullable=True),
    Column("delta_from_fx_pct", Numeric(8, 3), nullable=True),
    Column("coverage_gate_passed", Boolean, nullable=False),
    PrimaryKeyConstraint("hs6", "flow", "year"),
)

# [docs/PLAN.md §4, PM-1 fix: check B partner dimension] partner_country_code
# defaults to 'ALL_PARTNERS' for checks A/C (aggregate, not per-partner).
analytics_mismatch_checks = Table(
    "analytics_mismatch_checks",
    metadata,
    Column("hs6", Text, nullable=False),
    Column("flow", Text, nullable=False),
    Column("year", Integer, nullable=False),
    Column("check_name", Text, nullable=False),
    Column("partner_country_code", Text, nullable=False, server_default=text("'ALL_PARTNERS'")),
    Column("gap_pct", Numeric(8, 3), nullable=False),
    Column("severity", Text, nullable=False),
    Column("direction_flip_yoy", Boolean, nullable=False, server_default=text("false")),
    CheckConstraint(
        "check_name IN "
        "('A_dgcis_vs_comtrade_india','B_dgcis_vs_partner_comtrade','C_dgcis_vs_baci')",
        name="ck_amc_check_name",
    ),
    CheckConstraint(
        "severity IN ('quiet','flag','warning','untrustworthy')", name="ck_amc_severity"
    ),
    PrimaryKeyConstraint("hs6", "flow", "year", "check_name", "partner_country_code"),
)

# [docs/PLAN.md §4, PM-1 fix: mid-year duty-rate ambiguity] Monthly grain,
# not yearly — see landed_cost.py's "as of <month>" headline convention.
# [docs/PLAN.md §4a, 2026-08-23 evidence-first revision] Every duty-amount
# column is nullable and stays NULL when its component wasn't VERIFIED.
# landed_cost_inr_paise_per_kg is populated only when is_complete;
# partial_landed_cost_inr_paise_per_kg is always computable from whichever
# components are VERIFIED, explicitly separate so nothing ever confuses
# the two.
analytics_landed_cost = Table(
    "analytics_landed_cost",
    metadata,
    Column("hs8", Text, nullable=False),
    Column("month", Date, nullable=False),
    Column("cif_inr_paise_per_kg", BigInteger, nullable=False),
    Column("bcd_inr_paise_per_kg", BigInteger, nullable=True),
    Column("aidc_inr_paise_per_kg", BigInteger, nullable=True),
    Column("sws_inr_paise_per_kg", BigInteger, nullable=True),
    Column("igst_inr_paise_per_kg", BigInteger, nullable=True),
    Column("is_complete", Boolean, nullable=False),
    Column("excluded_components", ARRAY(Text), nullable=False, server_default=text("'{}'")),
    Column("landed_cost_inr_paise_per_kg", BigInteger, nullable=True),
    Column("partial_landed_cost_inr_paise_per_kg", BigInteger, nullable=True),
    Column("domestic_price_inr_paise_per_kg", BigInteger, nullable=True),
    Column("margin_pct", Numeric(8, 3), nullable=True),
    Column("domestic_price_confidence", Text, nullable=False),
    CheckConstraint(
        "domestic_price_confidence IN ('good','limited','unavailable')",
        name="ck_alc_domestic_price_confidence",
    ),
    PrimaryKeyConstraint("hs8", "month"),
)

analytics_coverage_summary = Table(
    "analytics_coverage_summary",
    metadata,
    Column("hs6", Text, nullable=False),
    Column("flow", Text, nullable=False),
    Column("window_start", Date, nullable=False),
    Column("window_end", Date, nullable=False),
    Column("expected_cells", Integer, nullable=False),
    Column("present_cells", Integer, nullable=False),
    Column("not_yet_published_cells", Integer, nullable=False),
    Column("suppressed_cells", Integer, nullable=False),
    Column("fetch_failed_cells", Integer, nullable=False),
    Column("gate_passed", Boolean, nullable=False),
    Column("degraded", Boolean, nullable=False),
    PrimaryKeyConstraint("hs6", "flow", "window_start", "window_end"),
)

__all__ = [
    "CELL_STATUS_VALUES",
    "DUTY_COMPONENTS",
    "DUTY_VERIFICATION_STATUS_VALUES",
    "analytics_coverage_summary",
    "analytics_landed_cost",
    "analytics_mismatch_checks",
    "analytics_monthly_current_year",
    "analytics_partner_rankings",
    "analytics_unit_value_series",
    "cell_status_enum",
    "dead_letter_ingestion",
    "duty_verification_status_enum",
    "metadata",
    "normalized_trade_flows",
    "raw_agmarknet_prices",
    "raw_baci_records",
    "raw_comtrade_records",
    "raw_dgcis_annual",
    "raw_dgcis_monthly",
    "ref_country_crosswalk",
    "ref_duty_component_conflicts",
    "ref_duty_components",
    "ref_hs6_hs8_crosswalk",
    "ref_hs_revision_notes",
    "ref_regulatory_notes",
]
