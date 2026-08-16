"""Evaluation runner (docs/PLAN.md §7 "Eval gate").

Scores every row of `evals/dataset.jsonl` on two axes, deliberately
different weight classes (docs/PLAN.md §7's table: "number-grounding
failure blocks merge, judge-score drift warns, doesn't block"):

1. **Number-grounding — deterministic, BLOCKING.** Runs the real
   `app.nodes.summarize.summarize` node (under `LLM_PROVIDER=mock`, so this
   is free and has zero flakiness) against a small synthetic-but-real
   `TradeTable` built via the real `app.nodes.aggregate.build_trade_table`,
   then asserts `app.guardrails.check_numbers_grounded` passes on its
   output. This is exactly the same enforcement path a live request goes
   through (`summarize` itself calls this guardrail and turns a failure
   into `ErrorResponse(error_code="UNGROUNDED_SUMMARY")` before this script
   ever sees it) — this eval is testing that the pipeline's own grounding
   enforcement stays wired up across changes, not testing model quality
   (there is no live model call here at all).

2. **Description-shape check — deterministic, WARN-ONLY.** Checks
   `description_must_reference`/`description_must_not_contain` as plain
   case-insensitive substring tests against `taxonomy_text` — the real,
   checked-in HS taxonomy source text for the row's `hs_code`
   (`app.knowledge.provider.build_taxonomy_text`) — NOT against
   `describe_item`'s output. This is a deliberate choice, not an
   oversight: this script also *calls* the real `describe_item` node (under
   `LLM_PROVIDER=mock`, so free) to prove that code path still runs
   end-to-end, but its output can't be content-checked here.
   `app.models.MockLLM`'s `_mock_text_for` (verified/frozen — see the
   README/task this script was written under) echoes numeric tokens from
   its input when any are present, and `describe_item`'s input always
   starts with the literal HS code digits (`f"HS code: {hs_code}\n\n..."`)
   — so its mock output is *always* the numeric-echo branch, never the
   taxonomy-text excerpt, making a content check against it fire on every
   single row regardless of dataset quality (verified by running this
   script during development: 100% warn rate, zero discriminating signal).
   Checking against `taxonomy_text` directly instead is honest about what's
   actually being verified: that this dataset's own reference phrases are
   accurate against the real official source text, not a proxy for model
   description quality (a real LLM-as-judge, deferred — docs/PLAN.md §7 —
   would be needed for that; explicitly optional for the task this script
   was written under: "if you add an LLM-as-judge check").

Every check's *dataset integrity* precondition (the row's `hs_code` really
is a known HS6 code; the dataset spans a real variety of sections, not one
commodity type) is its own always-blocking category, separate from the two
above — a bad eval dataset is a bug in the suite itself, not a soft signal.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

# Allow `python evals/run_evals.py` from any CWD to import `app.*` — a
# script invoked as a file path (not `python -m`) only gets its own
# directory on sys.path by default, not the repo root.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Safe placeholders for Settings' required fields, mirroring
# tests/conftest.py exactly — this script has no pytest session to rely on
# for that, and must still run standalone (`uv run python evals/run_evals.py`)
# with zero real secrets, zero token spend (master brief §6).
os.environ.setdefault("COMTRADE_API_KEY", "eval-placeholder-comtrade-key")
os.environ.setdefault("GEMINI_API_KEY", "eval-placeholder-gemini-key")
os.environ.setdefault("LLM_PROVIDER", "mock")
os.environ.setdefault("MODEL_UTILITY", "mock-utility")
os.environ.setdefault("MODEL_ANALYSIS", "mock-analysis")
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")
os.environ.setdefault("ENVIRONMENT", "test")

DATASET_PATH = Path(__file__).parent / "dataset.jsonl"
# docs/PLAN.md §7: "avoids a set that's accidentally all one commodity type"
MIN_DISTINCT_SECTIONS = 10

REQUIRED_ROW_FIELDS = {
    "hs_code",
    "section",
    "section_name",
    "description_must_reference",
    "description_must_not_contain",
}


def load_dataset(path: Path = DATASET_PATH) -> list[dict[str, Any]]:
    """Read and JSON-parse every line of the eval dataset."""
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            missing = REQUIRED_ROW_FIELDS - row.keys()
            if missing:
                raise ValueError(f"{path.name}:{line_no}: missing field(s) {sorted(missing)}")
            rows.append(row)
    return rows


def check_dataset_integrity(rows: list[dict[str, Any]]) -> list[str]:
    """Always-blocking checks on the dataset itself. Returns a list of
    failure messages (empty = dataset is internally sound)."""
    from app.knowledge.provider import is_known_hs6_code

    failures: list[str] = []
    for row in rows:
        if not is_known_hs6_code(row["hs_code"]):
            failures.append(
                f"{row['hs_code']!r} is not a real HS6 code in data/harmonized-system.csv"
            )

    distinct_sections = {row["section"] for row in rows}
    if len(distinct_sections) < MIN_DISTINCT_SECTIONS:
        failures.append(
            f"dataset spans only {len(distinct_sections)} distinct HS section(s) "
            f"(need >= {MIN_DISTINCT_SECTIONS}) — too narrow to be a meaningful eval set"
        )
    return failures


def _synthetic_table(hs_code: str, *, flow: Any, magnitude: int) -> Any:
    """A small, deterministic, per-row-varying `TradeTable` — real
    aggregation code (`build_trade_table`), synthetic input records. Values
    scale with `magnitude` (the row's position in the dataset) purely so
    different rows don't all produce visually-identical numbers; nothing
    about the grounding check depends on the specific values chosen."""
    from app.nodes.aggregate import build_trade_table
    from app.tools.comtrade_client import ComtradeRecord

    records = [
        ComtradeRecord(
            hs_code=hs_code,
            flow=flow,
            partner_code="842",
            partner_country="USA",
            year=2023,
            value=1000.0 * magnitude,
            is_provisional=False,
        ),
        ComtradeRecord(
            hs_code=hs_code,
            flow=flow,
            partner_code="156",
            partner_country="China",
            year=2023,
            value=500.0 * magnitude,
            is_provisional=False,
        ),
        ComtradeRecord(
            hs_code=hs_code,
            flow=flow,
            partner_code="842",
            partner_country="USA",
            year=2024,
            value=1100.0 * magnitude,
            is_provisional=True,
        ),
    ]
    return build_trade_table(records, years=[2023, 2024])


async def _score_row(row: dict[str, Any], *, index: int) -> tuple[list[str], list[str]]:
    """Returns `(blocking_failures, warnings)` for one dataset row."""
    from app.guardrails import check_numbers_grounded
    from app.knowledge.provider import build_taxonomy_text
    from app.nodes.describe_item import describe_item
    from app.nodes.summarize import summarize
    from app.schemas.query import TradeQuery

    hs_code = row["hs_code"]
    blocking: list[str] = []
    warnings: list[str] = []
    thread_id = f"eval-{hs_code}"  # unique per row: exactly 2 model calls/thread, at the ceiling

    # --- Description shape (warn-only, checked against the real taxonomy
    # source text — see this module's docstring for why not against
    # describe_item's own output) ---------------------------------------------
    taxonomy_text = build_taxonomy_text(hs_code)
    assert taxonomy_text is not None  # dataset integrity already checked this hs_code is real
    lowered_source = taxonomy_text.lower()
    for phrase in row["description_must_reference"]:
        if phrase.lower() not in lowered_source:
            warnings.append(f"{hs_code}: taxonomy source text did not contain {phrase!r}")
    for phrase in row["description_must_not_contain"]:
        if phrase.lower() in lowered_source:
            warnings.append(f"{hs_code}: taxonomy source text unexpectedly contained {phrase!r}")

    # Still exercise the real describe_item node end-to-end (schema-
    # constrained MockLLM call, real prompt loading, real budget check) even
    # though its mock output can't be content-checked (see docstring) — a
    # regression there (e.g. a raised exception) should still fail the eval.
    describe_state = {
        "query": TradeQuery(hs_code=hs_code),
        "taxonomy_text": taxonomy_text,
        "thread_id": thread_id,
    }
    describe_result = await describe_item(describe_state)  # type: ignore[arg-type]
    if "item_description" not in describe_result:
        blocking.append(f"{hs_code}: describe_item did not produce item_description")

    # --- Number-grounding (BLOCKING) -----------------------------------------
    imports_table = _synthetic_table(hs_code, flow="import", magnitude=index + 1)
    exports_table = _synthetic_table(hs_code, flow="export", magnitude=index + 1)
    summarize_state = {
        "query": TradeQuery(hs_code=hs_code),
        "imports_table": imports_table,
        "exports_table": exports_table,
        "thread_id": thread_id,
    }
    summarize_result = await summarize(summarize_state)  # type: ignore[arg-type]
    if "error" in summarize_result:
        blocking.append(
            f"{hs_code}: summarize produced an error instead of a summary: "
            f"{summarize_result['error'].error_code}"
        )
    else:
        summary = summarize_result["analytical_summary"]
        if not check_numbers_grounded(summary, imports_table, exports_table, hs_code=hs_code):
            from app.guardrails import find_ungrounded_numbers

            offending = find_ungrounded_numbers(
                summary, imports_table, exports_table, hs_code=hs_code
            )
            blocking.append(f"{hs_code}: ungrounded number(s) in summary: {offending}")

    return blocking, warnings


async def run() -> int:
    rows = load_dataset()
    print(f"run_evals.py: loaded {len(rows)} row(s) from {DATASET_PATH.name}")

    integrity_failures = check_dataset_integrity(rows)
    if integrity_failures:
        print("\nDATASET INTEGRITY FAILURES (always blocking):")
        for failure in integrity_failures:
            print(f"  - {failure}")
        return 1

    all_blocking: list[str] = []
    all_warnings: list[str] = []
    for index, row in enumerate(rows):
        blocking, warnings = await _score_row(row, index=index)
        all_blocking.extend(blocking)
        all_warnings.extend(warnings)

    print(f"\nScored {len(rows)} row(s) across {len({r['section'] for r in rows})} HS sections.")

    if all_warnings:
        print(f"\nWARNINGS ({len(all_warnings)}, non-blocking — description-shape check):")
        for warning in all_warnings:
            print(f"  - {warning}")

    if all_blocking:
        print(f"\nBLOCKING FAILURES ({len(all_blocking)} — number-grounding):")
        for failure in all_blocking:
            print(f"  - {failure}")
        print("\neval-gate: FAILED (number-grounding regression)")
        return 1

    print("\neval-gate: PASSED (number-grounding: 0 failures)")
    return 0


def main() -> int:
    return asyncio.run(run())


if __name__ == "__main__":
    sys.exit(main())
