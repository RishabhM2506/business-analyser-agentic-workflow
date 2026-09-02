"""Process-wide async engine for the trade pipeline's warehouse tables
(`app.warehouse.schema`). Matches this repo's existing singleton pattern
for shared, expensive-to-construct resources (`app.budget.get_budget_tracker`,
`app.cache.tool_cache.get_tool_cache`) and `app.main.check_database`'s
exact `create_async_engine(url, pool_pre_ping=True)` call shape.

Separate from LangGraph's own checkpointer engine (`app/graph.py`) even
though both may point at the same Postgres instance in `docker-compose.yml`
— the checkpointer manages its own connection lifecycle internally
(`build_checkpointer`), and this module has no reason to share it.
"""

from __future__ import annotations

from functools import lru_cache

from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from app.settings import get_settings


@lru_cache(maxsize=1)
def get_engine() -> AsyncEngine:
    """Return the process-wide `AsyncEngine` for the warehouse tables,
    constructed on first use from `settings.database_url`."""
    return create_async_engine(get_settings().database_url, pool_pre_ping=True)
