"""ref_llm_datapoints table

Revision ID: 9aab143535b9
Revises: 9bfa1484a1d0
Create Date: 2026-09-02 22:53:37.131363

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '9aab143535b9'
down_revision: Union[str, Sequence[str], None] = '9bfa1484a1d0'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Created explicitly, once, before the table references it — same
    # DO-block idiom as duty_verification_status (14691feaef80) and
    # cell_status (2b3646342c9b), for the same reason those use it over
    # SQLAlchemy's checkfirst.
    op.execute(
        """
        DO $$ BEGIN
            CREATE TYPE llm_datapoint_status AS ENUM ('ACTIVE', 'RETRACTED');
        EXCEPTION WHEN duplicate_object THEN null;
        END $$;
        """
    )
    llm_datapoint_status_enum = postgresql.ENUM(
        'ACTIVE', 'RETRACTED',
        name='llm_datapoint_status',
        create_type=False,
    )

    op.create_table('ref_llm_datapoints',
    sa.Column('id', sa.BigInteger(), autoincrement=True, nullable=False),
    sa.Column('hs6', sa.Text(), nullable=False),
    sa.Column('field_name', sa.Text(), nullable=False),
    sa.Column('effective_period', sa.Text(), nullable=False),
    sa.Column('value_json', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
    sa.Column('source_authority', sa.Text(), nullable=False),
    sa.Column('source_reference', sa.Text(), nullable=False),
    sa.Column('source_url', sa.Text(), nullable=True),
    sa.Column('verified_date', sa.Date(), nullable=False),
    sa.Column('status', llm_datapoint_status_enum, server_default=sa.text("'ACTIVE'"), nullable=False),
    sa.Column('retracted_reason', sa.Text(), nullable=True),
    sa.Column('notes', sa.Text(), nullable=True),
    sa.CheckConstraint("(status = 'ACTIVE' AND retracted_reason IS NULL) OR (status = 'RETRACTED' AND retracted_reason IS NOT NULL)", name='ck_rld_retracted_reason_matches_status'),
    sa.CheckConstraint("field_name IN ('mandi_price','msp','international_production')", name='ck_rld_field_name'),
    sa.PrimaryKeyConstraint('id')
    )
    # NOTE: autogenerate also detected the checkpoint_* tables as "removed"
    # (LangGraph's own tables, not ours — see the matching note in earlier
    # migrations, e.g. 14691feaef80). Stripped by hand, same as before.


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table('ref_llm_datapoints')
    postgresql.ENUM(name='llm_datapoint_status').drop(op.get_bind(), checkfirst=True)
