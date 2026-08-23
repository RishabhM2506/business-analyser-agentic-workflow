"""normalized_trade_flows unique constraint nulls not distinct

Revision ID: 971cfa94fae9
Revises: 85c781cf7550
Create Date: 2026-08-23 12:49:04.859488

Real bug found live: Comtrade-sourced normalized_trade_flows rows always
have hs8=NULL (Comtrade is HS6-level only), and standard SQL treats NULL
as distinct from NULL even inside a unique constraint - so ON CONFLICT
never matched two otherwise-identical Comtrade rows, and every
re-normalization run silently duplicated the entire Comtrade slice
instead of upserting in place (confirmed live: 363 real rows had grown to
726 after a single extra re-run). Postgres 15+'s NULLS NOT DISTINCT makes
NULL compare equal to NULL for this constraint only.

(Autogenerate's LangGraph checkpoint-table noise - this database also
hosts the app's own checkpointer tables - stripped by hand, per this
migration history's established pattern.)
"""
from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = '971cfa94fae9'
down_revision: Union[str, Sequence[str], None] = '85c781cf7550'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.drop_constraint(op.f('uq_normalized_trade_flows'), 'normalized_trade_flows', type_='unique')
    op.create_unique_constraint('uq_normalized_trade_flows', 'normalized_trade_flows', ['source', 'hs6', 'hs8', 'flow', 'period_month', 'partner_country_code', 'dataset_version'], postgresql_nulls_not_distinct=True)


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_constraint('uq_normalized_trade_flows', 'normalized_trade_flows', type_='unique')
    op.create_unique_constraint(op.f('uq_normalized_trade_flows'), 'normalized_trade_flows', ['source', 'hs6', 'hs8', 'flow', 'period_month', 'partner_country_code', 'dataset_version'], postgresql_nulls_not_distinct=False)
