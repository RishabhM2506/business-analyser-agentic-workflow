CURRENT_STEP: 3
CURRENT_ROLE: Software engineer (backend)
ITERATION: 1
GATE_STATUS: in_progress
LAST_COMPLETED: Build sequence item 4's annual per-partner DGCIS path complete and verified end to end (client, parser, per-country loop, idempotent upsert, real data in the local Postgres), plus the canonical scenario's HS8-split check resolved live (120791 has exactly one ITC-HS8 child). Build sequence item 5 (Comtrade D5 bulk-batching) fully verified live using the real Comtrade key: every dimension (period, flowCode, reporterCode-omitted, partnerCode-omitted, and a bonus — cmdCode itself) batches correctly, confirmed individually and combined (265 rows in one call). No fallback needed; docs/PLAN.md's two-query design is confirmed as originally written. 393 tests passing, mypy/ruff/black clean.
NEXT_TASK: Build app/pipeline/comtrade_mirror.py against the now-fully-confirmed design — the D6 fixed retry schedule ([30,60,120,300]s, distinct from the existing single-lookup ComtradeClient's exponential backoff), the two query shapes (Query 1: reporterCode=699 for check A; Query 2: partnerCode=699 for check B), circuit breaker (3 consecutive 429s -> 15min pause), and the raw_comtrade_records upsert. Then real live smoke test with the poppy-seed HS6.
BLOCKED_ON: none — buildable now. Still open, not currently blocking: no real HS 120791 duty rates recorded yet (mechanism built and verified; citation still pending from the user); Agmarknet data.gov.in API key (blocks item 7); DGCIS's monthly national-total path (D15) and full ~250-country rate-limit tuning still open within item 4.
OPEN_FINDINGS: none open
FILES_IN_FLIGHT: none
