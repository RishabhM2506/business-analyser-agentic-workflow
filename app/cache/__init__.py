"""Cache layers (docs/PLAN.md §5.4): `response_cache` (application-level,
full response) and `tool_cache` (raw Comtrade fetches). No Redis in v1 —
in-process/SQLite-backed, single-instance (docs/PLAN.md §1.2).
"""
