"""Evaluation runner (docs/PLAN.md §7 "Eval gate").

Phase 2 scope: read `evals/dataset.jsonl` and confirm it parses into the
expected shape. No scoring yet — number-grounding scoring and the
LLM-as-judge description check both depend on code that doesn't exist until
Phase 3 (`app/guardrails.py`, `app/nodes/describe_item.py`).

# TODO(Phase 3): score each row for (a) deterministic number-grounding via
# `app.guardrails.check_numbers_grounded` and (b) description-accuracy via
# an LLM-as-judge, cassette-replayed in CI per docs/PLAN.md §7. Wire this
# into the `eval-gate` CI job once real scoring exists — see README.md for
# why that job isn't in Phase 2's CI workflow yet.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

DATASET_PATH = Path(__file__).parent / "dataset.jsonl"


def load_dataset(path: Path = DATASET_PATH) -> list[dict[str, Any]]:
    """Read and JSON-parse every line of the eval dataset."""
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def main() -> int:
    rows = load_dataset()
    print(f"run_evals.py: loaded {len(rows)} row(s) from {DATASET_PATH.name} (Phase 2: no-op).")
    for row in rows:
        assert "hs_code" in row, f"malformed eval row, missing hs_code: {row}"
    return 0


if __name__ == "__main__":
    sys.exit(main())
