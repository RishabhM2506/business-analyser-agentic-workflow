<!--
PROMPT_VERSION = "v0-scaffold"

Phase 2 placeholder. Real prompt content — the MODEL_ANALYSIS system prompt
used by app/nodes/summarize.py — is Phase 3 work (docs/PLAN.md §5.1,
§5.6). PROMPT_VERSION is included in every trace's metadata
(app/observability.py) and in the response-cache key
(app/cache/response_cache.py) — bumping it is a deliberate cache-busting
event, not a silent behavior change against stale cached output.
-->

# summarize — placeholder

Not yet written. See docs/PLAN.md §5.1 (call #2: MODEL_ANALYSIS, ~1,100
input / ~200 output tokens over the pre-aggregated table only — never raw
Comtrade JSON) and §5.6 (prompt storage/versioning convention).
