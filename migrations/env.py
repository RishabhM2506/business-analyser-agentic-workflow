import asyncio
import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

from app.warehouse.schema import metadata as target_metadata

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# `target_metadata` (imported above) is `app.warehouse.schema.metadata` —
# the single source of truth (docs/PLAN.md §4) — every ingestion/report
# module imports the same Table objects, so schema and code can never
# drift apart via a second, hand-maintained DDL file.

# `sqlalchemy.url` in alembic.ini is a placeholder — the real URL always
# comes from the `DATABASE_URL` environment variable, read directly here
# (not via `app.settings.get_settings()`) — live-verified running this
# inside the built container: `get_settings()` requires
# `comtrade_api_key`/`gemini_api_key` too (unrelated Settings fields for
# the existing, separate trade-analysis feature), which would make running
# migrations alone fail on missing credentials that have nothing to do
# with a database connection. Same default as `Settings.database_url`
# for parity with the rest of the app.
if not config.get_main_option("sqlalchemy.url", "").strip():
    config.set_main_option(
        "sqlalchemy.url", os.environ.get("DATABASE_URL", "sqlite+aiosqlite:///./local.db")
    )


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.

    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    context.configure(connection=connection, target_metadata=target_metadata)

    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """In this scenario we need to create an Engine
    and associate a connection with the context.

    """

    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode."""

    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
