"""raw_msp_records

Revision ID: 9d9b27ce34f8
Revises: 971cfa94fae9
Create Date: 2026-08-24 04:19:28.258353

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "9d9b27ce34f8"
down_revision: Union[str, Sequence[str], None] = "971cfa94fae9"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "raw_msp_records",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("fetched_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("crops", sa.Text(), nullable=False),
        sa.Column("commodity", sa.Text(), nullable=False),
        sa.Column("year_label", sa.Text(), nullable=False),
        sa.Column("cost_inr_paise_per_qtl", sa.BigInteger(), nullable=True),
        sa.Column("msp_inr_paise_per_qtl", sa.BigInteger(), nullable=True),
        sa.Column("raw_payload", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("commodity", "year_label", name="uq_raw_msp_records"),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table("raw_msp_records")
