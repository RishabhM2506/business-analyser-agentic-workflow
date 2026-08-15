"""`KnowledgeProvider` interface + `StaticKnowledgeProvider` (v1 impl).

Named seam for the future vector-DB/agentic-RAG knowledge layer (master
brief §3: "A `KnowledgeProvider` interface with a v1 no-op/static
implementation. The graph already has a `retrieve` node that currently
returns a static description."). v1's implementation is a lookup against
the checked-in `data/harmonized-system.csv` taxonomy — no vector store, no
embeddings, no ingestion of untrusted text (docs/PLAN.md §1.2, §6).

# TODO(Phase 3): implement `StaticKnowledgeProvider.retrieve()` as a lookup
# keyed by `hs_code` against `data/harmonized-system.csv`.
"""

from __future__ import annotations

from typing import Protocol


class KnowledgeProvider(Protocol):
    """Interface every knowledge-retrieval implementation must satisfy."""

    async def retrieve(self, hs_code: str) -> str:
        """Return descriptive source text for an HS6 code."""
        ...


class StaticKnowledgeProvider:
    """v1 implementation: static lookup against the checked-in taxonomy CSV."""

    def __init__(self, csv_path: str = "data/harmonized-system.csv") -> None:
        self._csv_path = csv_path

    async def retrieve(self, hs_code: str) -> str:
        raise NotImplementedError  # TODO(Phase 3): implement CSV lookup keyed by hs_code.
