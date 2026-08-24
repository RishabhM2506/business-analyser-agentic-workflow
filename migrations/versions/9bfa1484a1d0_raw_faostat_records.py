"""raw_faostat_records

Revision ID: 9bfa1484a1d0
Revises: 9d9b27ce34f8
Create Date: 2026-08-25 00:34:54.956733

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "9bfa1484a1d0"
down_revision: str | Sequence[str] | None = "9d9b27ce34f8"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "raw_faostat_records",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("fetched_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("area_code", sa.Text(), nullable=False),
        sa.Column("area", sa.Text(), nullable=False),
        sa.Column("item_code", sa.Text(), nullable=False),
        sa.Column("item", sa.Text(), nullable=False),
        sa.Column("element", sa.Text(), nullable=False),
        sa.Column("unit", sa.Text(), nullable=False),
        sa.Column("year", sa.Integer(), nullable=False),
        sa.Column("value", sa.Numeric(precision=18, scale=3), nullable=True),
        sa.Column("flag", sa.Text(), nullable=True),
        sa.Column("raw_payload", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "area_code", "item_code", "element", "year", name="uq_raw_faostat_records"
        ),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table("raw_faostat_records")
