"""Unit tests for `app.main`'s pure, no-I/O pieces: the `error_code` ->
HTTP status map. Pure lookup, no I/O — `unit`, not `integration` (the full
request/response behavior around these is covered by
`tests/integration/test_threads_api.py`).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from app.main import _ERROR_STATUS_CODES, _status_code_for_error

_MAIN_PY_SOURCE = (Path(__file__).resolve().parents[2] / "app" / "main.py").read_text(
    encoding="utf-8"
)


@pytest.mark.unit
def test_thread_incomplete_maps_to_404() -> None:
    """Finding M3 (architect review, 2026-08-20): `THREAD_INCOMPLETE` was
    constructed with a hardcoded `status_code=404` at its one call site
    (`GET /threads/{id}`), bypassing this map entirely — a real drift
    against this module's own docstring, which claims every producible
    error code is listed here explicitly. Harmless today only because the
    hardcoded value happened to already agree with what a correct map
    entry would say; this test pins that agreement so it can't silently
    drift the next time either the call site or this map is touched."""
    assert _ERROR_STATUS_CODES["THREAD_INCOMPLETE"] == 404
    assert _status_code_for_error("THREAD_INCOMPLETE") == 404


@pytest.mark.unit
def test_error_status_codes_map_is_genuinely_exhaustive() -> None:
    """Mechanically enforces this module's own docstring claim ("every code
    any node/endpoint can currently produce is listed explicitly") by
    scanning the real source for every `error_code="..."` literal
    constructed anywhere in `app/main.py` and asserting each one is a key
    in `_ERROR_STATUS_CODES`. This is exactly the check that would have
    caught M3 automatically before a live request ever exercised the
    THREAD_INCOMPLETE path."""
    constructed_codes = set(re.findall(r'error_code="([A-Z_]+)"', _MAIN_PY_SOURCE))
    assert constructed_codes, 'regex found no error_code="..." literals — pattern likely stale'
    missing = constructed_codes - set(_ERROR_STATUS_CODES)
    assert (
        not missing
    ), f"error code(s) constructed in app/main.py but missing from the map: {missing}"


@pytest.mark.unit
def test_unrecognized_error_code_falls_back_to_500() -> None:
    assert _status_code_for_error("SOME_CODE_NOT_IN_THE_MAP") == 500
