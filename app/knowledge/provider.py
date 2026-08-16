"""`KnowledgeProvider` interface + `StaticKnowledgeProvider` (v1 impl).

Named seam for the future vector-DB/agentic-RAG knowledge layer (master
brief §3: "A `KnowledgeProvider` interface with a v1 no-op/static
implementation. The graph already has a `retrieve` node that currently
returns a static description."). v1's implementation is a lookup against
the checked-in `data/harmonized-system.csv` taxonomy — no vector store, no
embeddings, no ingestion of untrusted text (docs/PLAN.md §1.2, §6).

Also the single source of truth for taxonomy lookups used elsewhere:
`app.guardrails.check_hs_code_allowlisted` (the input-guardrail allowlist
check, docs/PLAN.md §6) delegates to `is_known_hs6_code` below rather than
re-parsing the CSV — one loader, one cache, two callers.

`data/harmonized-system-sections.csv` (the 21 official WCO HS section
names, e.g. "Section I: live animals; animal products") is vendored
alongside the taxonomy from the same source repo
(`github.com/datasets/harmonized-system`, ODC-PDDL public domain — the
exact Gate-0-approved source) and fetched live the same day as the rest of
Phase 3's research, used only to give `describe_item`'s prompt slightly
richer, still 100%-official-source grounding — never invented.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Protocol

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_TAXONOMY_PATH = "data/harmonized-system.csv"
_DEFAULT_SECTIONS_PATH = "data/harmonized-system-sections.csv"
HS6_LEVEL = "6"


def _resolve_path(csv_path: str) -> Path:
    """Resolve a (possibly relative, matching the friendly documented
    default) path against the repo root rather than the process's current
    working directory — robust regardless of where the process was
    launched from (pytest, docker WORKDIR, a deploy script, ...)."""
    path = Path(csv_path)
    return path if path.is_absolute() else _REPO_ROOT / path


@dataclass(frozen=True)
class TaxonomyEntry:
    """One row of the checked-in HS taxonomy CSV."""

    hs_code: str
    description: str
    section: str
    parent: str
    level: str


class KnowledgeProvider(Protocol):
    """Interface every knowledge-retrieval implementation must satisfy."""

    async def retrieve(self, hs_code: str) -> str:
        """Return descriptive source text for an HS6 code."""
        ...


@lru_cache(maxsize=4)
def _load_taxonomy(resolved_path: Path) -> dict[str, TaxonomyEntry]:
    entries: dict[str, TaxonomyEntry] = {}
    with resolved_path.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            entries[row["hscode"]] = TaxonomyEntry(
                hs_code=row["hscode"],
                description=row["description"],
                section=row["section"],
                parent=row["parent"],
                level=row["level"],
            )
    return entries


@lru_cache(maxsize=4)
def _load_section_names(resolved_path: Path) -> dict[str, str]:
    names: dict[str, str] = {}
    with resolved_path.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            names[row["section"]] = row["name"]
    return names


def get_taxonomy_entry(
    hs_code: str, *, taxonomy_path: str = _DEFAULT_TAXONOMY_PATH
) -> TaxonomyEntry | None:
    """Look up one HS code's raw taxonomy row (any level, not just HS6)."""
    return _load_taxonomy(_resolve_path(taxonomy_path)).get(hs_code)


def is_known_hs6_code(hs_code: str, *, taxonomy_path: str = _DEFAULT_TAXONOMY_PATH) -> bool:
    """True iff `hs_code` is a real HS6 sub-heading (taxonomy `level == "6"`)
    in the checked-in taxonomy — the input-guardrail allowlist check
    (docs/PLAN.md §6: "an invalid or adversarial code never reaches the
    Comtrade client or a model")."""
    entry = get_taxonomy_entry(hs_code, taxonomy_path=taxonomy_path)
    return entry is not None and entry.level == HS6_LEVEL


def build_taxonomy_text(
    hs_code: str,
    *,
    taxonomy_path: str = _DEFAULT_TAXONOMY_PATH,
    sections_path: str = _DEFAULT_SECTIONS_PATH,
) -> str | None:
    """Compose the official-source text `describe_item` writes prose about:
    the HS6 description, its parent-heading breadcrumb, and its section
    name — entirely from the checked-in taxonomy CSVs, nothing invented.
    Returns `None` if `hs_code` isn't a known HS6 code (callers allowlist-
    check first via `is_known_hs6_code`; `StaticKnowledgeProvider.retrieve`
    below raises rather than returning `None`)."""
    entries = _load_taxonomy(_resolve_path(taxonomy_path))
    entry = entries.get(hs_code)
    if entry is None or entry.level != HS6_LEVEL:
        return None

    breadcrumb: list[str] = [entry.description]
    seen = {hs_code}
    parent = entries.get(entry.parent)
    while parent is not None and parent.hs_code not in seen:
        breadcrumb.append(parent.description)
        seen.add(parent.hs_code)
        parent = entries.get(parent.parent)

    lines = [f"HS {hs_code}: {entry.description}"]
    if len(breadcrumb) > 1:
        ancestry = " > ".join(reversed(breadcrumb[1:]))
        lines.append(f"Parent headings: {ancestry}")

    section_name = _load_section_names(_resolve_path(sections_path)).get(entry.section)
    if section_name:
        lines.append(f"HS Section {entry.section}: {section_name[0].upper()}{section_name[1:]}")

    return "\n".join(lines)


class StaticKnowledgeProvider:
    """v1 implementation: static lookup against the checked-in taxonomy CSV."""

    def __init__(self, csv_path: str = _DEFAULT_TAXONOMY_PATH) -> None:
        self._csv_path = csv_path

    async def retrieve(self, hs_code: str) -> str:
        text = build_taxonomy_text(hs_code, taxonomy_path=self._csv_path)
        if text is None:
            raise KeyError(f"hs_code {hs_code!r} is not a known HS6 code in {self._csv_path}")
        return text
