"""Cassette record/replay mechanism for `llm`-marked tests (docs/PLAN.md
§7's `llm`-marked test layer; master brief §6: "Record/replay for model
calls (VCR-style cassettes) so model-dependent tests are deterministic and
free after first record").

**No real Gemini API key exists yet** (the user is providing one
separately — see `docs/OPEN-QUESTIONS.md`), so unlike
`tests/fixtures/comtrade/` (which has real, live-captured UN Comtrade
responses because that API has a no-auth preview endpoint), every cassette
under `tests/llm/cassettes/` is **synthetic** — hand-authored,
structurally/schema-correct placeholder content, not a real recorded
Gemini response. Every cassette's top-level `_meta.synthetic` field must be
`true`; `CassetteModelClient` refuses to load a cassette missing that flag,
so a real recording can never be silently mistaken for a synthetic one or
vice versa.

**Re-recording against a real key.** Once a real `GEMINI_API_KEY` exists,
replace a cassette's `output` field with the real model's structured
response (keep `_meta.synthetic = false` and add `_meta.recorded_at`), or
extend `record_cassette` below into a small opt-in script gated behind an
explicit env var (e.g. `RECORD_CASSETTES=1`) — deliberately not wired into
any test or CI path yet, since doing so would make a test's pass/fail
depend on network+credentials, exactly what cassettes exist to avoid.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

CASSETTES_DIR = Path(__file__).parent / "cassettes"


class CassetteError(Exception):
    """Raised when a cassette file is missing, malformed, or missing the
    required `_meta.synthetic` marker."""


class CassetteModelClient:
    """Implements the same shape as `app.models.ModelClient` (a structural
    `Protocol` — no inheritance needed) by replaying a recorded structured-
    output payload from a JSON cassette file instead of calling a real
    model. Deterministic and free, every run — the entire point of a
    cassette."""

    def __init__(self, cassette_path: Path) -> None:
        self._cassette_path = cassette_path

    async def generate_structured(
        self, *, system_prompt: str, user_content: str, schema: type[T]
    ) -> T:
        if not self._cassette_path.exists():
            raise CassetteError(f"no cassette recorded at {self._cassette_path}")
        payload = json.loads(self._cassette_path.read_text(encoding="utf-8"))
        meta = payload.get("_meta", {})
        if "synthetic" not in meta:
            raise CassetteError(
                f"{self._cassette_path} is missing required `_meta.synthetic` — every cassette "
                "must explicitly mark itself real or synthetic, see this module's docstring"
            )
        if "output" not in payload:
            raise CassetteError(f"{self._cassette_path} has no `output` field to replay")
        return schema.model_validate(payload["output"])


def write_synthetic_cassette(
    path: Path, *, output: dict[str, object], note: str, **extra_meta: object
) -> None:
    """Helper used to author synthetic cassette fixtures (called from a
    one-off script / this test module's own setup, not from application
    code) — keeps the `_meta.synthetic = true` contract in one place rather
    than hand-typed separately in every cassette file."""
    payload = {
        "_meta": {"synthetic": True, "note": note, **extra_meta},
        "output": output,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
