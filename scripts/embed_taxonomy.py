"""One-time, developer-run script: embeds every HS6 taxonomy description via
the real Gemini embeddings API and writes the three files
`app.search.vector_index` loads at request time.

Never run in CI or at Docker build time — lives outside `app/`/`prompts/`/
`data/`'s Dockerfile COPY scope by construction (verified against the real
`Dockerfile`), and real embeddings cost real API spend. This corpus only
changes when the taxonomy CSV is re-vendored, so this is a rare, deliberate,
developer-run operation, not something any deploy pipeline invokes.

Usage (from the repo root, with a real `GEMINI_API_KEY` configured in
`.env` or the environment):

    uv run python scripts/embed_taxonomy.py

Rewrites all three `data/hs_taxonomy_embeddings.*` files in place on a
successful full run. Commit the result.

**Resumable across runs** (live-reproduced 2026-08-20: the free-tier Gemini
API key this was built against has a hard `1000 embed_content requests/day`
ceiling — `EmbedContentRequestsPerDayPerUserPerProjectPerModel-FreeTier` —
which the full 5,613-row corpus at 100 texts/batch exceeds in under 6 hours
of real-time-throttled batches; a real multi-day run is the normal case, not
a failure mode). Progress is written to
`data/.hs_taxonomy_embeddings.progress.{npy,json}` after every batch
(gitignored, never committed) and resumed automatically on the next
invocation, keyed on `source_csv_sha256` so an edited taxonomy never
silently resumes against stale embeddings for different text. A batch that
exhausts its own retries (e.g. the daily quota wall) prints a clear resume
message and exits — already-embedded progress is never lost.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from tenacity import AsyncRetrying, stop_after_attempt, wait_exponential_jitter

_REPO_ROOT = Path(__file__).resolve().parents[1]
# This script lives outside `app/`'s package tree on purpose (see module
# docstring) - Python doesn't put a plain script's own repo root on
# `sys.path` for absolute `app.*` imports, so it's added explicitly here,
# once, before those imports.
sys.path.insert(0, str(_REPO_ROOT))

from app.knowledge.provider import get_hs6_taxonomy_entries  # noqa: E402
from app.search.embeddings import GeminiEmbeddingsClient  # noqa: E402
from app.settings import get_settings  # noqa: E402

_OUTPUT_DIR = _REPO_ROOT / "data"
_OUTPUT_STEM = "hs_taxonomy_embeddings"
_TAXONOMY_CSV = _REPO_ROOT / "data" / "harmonized-system.csv"
_TASK_TYPE = "RETRIEVAL_DOCUMENT"

_PROGRESS_NPY_PATH = _OUTPUT_DIR / ".hs_taxonomy_embeddings.progress.npy"
_PROGRESS_META_PATH = _OUTPUT_DIR / ".hs_taxonomy_embeddings.progress.json"

# Google's own documented per-request ceiling, and (live-reproduced,
# 2026-08-20, against the real configured key) also roughly the entire
# free-tier "embed_content requests per minute" quota in one call: a batch
# of 100 texts alone triggered a 429 on the very next call sent within the
# same ~60s window.
_BATCH_SIZE = 100
# One saturating batch per this interval keeps each call inside its own
# fresh per-minute quota window instead of immediately re-hitting 429 the
# moment the next batch fires — comfortably above the 60s window with
# headroom for clock skew between this process and Google's quota counter.
_INTER_BATCH_DELAY_SECONDS = 65.0


def _sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_progress(*, source_csv_sha256: str) -> list[list[float]]:
    """Return whatever embeddings a previous, incomplete run already
    produced for *this exact* taxonomy CSV, or `[]` if there's no usable
    checkpoint (none exists, or it was written against a since-edited CSV —
    resuming against the wrong source text would silently corrupt the
    row<->hs_code alignment `app.search.vector_index` depends on)."""
    if not (_PROGRESS_NPY_PATH.exists() and _PROGRESS_META_PATH.exists()):
        return []
    meta = json.loads(_PROGRESS_META_PATH.read_text(encoding="utf-8"))
    if meta.get("source_csv_sha256") != source_csv_sha256:
        print(
            "Found a progress checkpoint, but it was written against a different "
            "taxonomy CSV (source_csv_sha256 mismatch) — ignoring it and starting fresh."
        )
        return []
    vectors: list[list[float]] = np.load(_PROGRESS_NPY_PATH).tolist()
    print(f"Resuming from checkpoint: {len(vectors)} texts already embedded.")
    return vectors


def _save_progress(vectors: list[list[float]], *, source_csv_sha256: str) -> None:
    np.save(_PROGRESS_NPY_PATH, np.asarray(vectors, dtype=np.float32))
    _PROGRESS_META_PATH.write_text(
        json.dumps({"count": len(vectors), "source_csv_sha256": source_csv_sha256}),
        encoding="utf-8",
    )


def _clear_progress() -> None:
    _PROGRESS_NPY_PATH.unlink(missing_ok=True)
    _PROGRESS_META_PATH.unlink(missing_ok=True)


async def _embed_remaining(
    client: GeminiEmbeddingsClient,
    texts: list[str],
    *,
    already_embedded: list[list[float]],
    source_csv_sha256: str,
) -> list[list[float]] | None:
    """Embed whatever of `texts` isn't already covered by
    `already_embedded`, in `_BATCH_SIZE` chunks, throttled to one batch per
    `_INTER_BATCH_DELAY_SECONDS` and retried with exponential backoff+jitter
    per batch. Saves progress after every batch. Returns the full vector
    list on success, or `None` if a batch's retries were exhausted (the
    caller should exit; progress up to that point is already on disk)."""
    vectors = list(already_embedded)
    remaining = texts[len(vectors) :]
    if not remaining:
        return vectors

    total_batches = (len(remaining) + _BATCH_SIZE - 1) // _BATCH_SIZE
    for batch_index in range(total_batches):
        batch = remaining[batch_index * _BATCH_SIZE : (batch_index + 1) * _BATCH_SIZE]
        retrying = AsyncRetrying(
            stop=stop_after_attempt(3),
            wait=wait_exponential_jitter(initial=_INTER_BATCH_DELAY_SECONDS, max=180),
            reraise=True,
        )
        try:
            batch_vectors: list[list[float]] = await retrying(client.embed_documents, batch)
        except Exception as exc:
            # Deliberately broad: any failure here must still preserve
            # progress and exit cleanly, not crash with a raw traceback
            # mid-run (a quota wall is the expected common case, per this
            # module's docstring, not an exceptional one).
            print(
                f"\nBatch {batch_index + 1}/{total_batches} failed after retries: {exc}\n"
                f"Progress saved: {len(vectors)}/{len(texts)} texts embedded so far.\n"
                "Re-run this script later (e.g. after the daily quota resets) to resume "
                "from exactly this point."
            )
            return None

        vectors.extend(batch_vectors)
        _save_progress(vectors, source_csv_sha256=source_csv_sha256)
        print(
            f"  batch {batch_index + 1}/{total_batches} embedded "
            f"({len(vectors)}/{len(texts)} total)"
        )
        if batch_index < total_batches - 1:
            await asyncio.sleep(_INTER_BATCH_DELAY_SECONDS)
    return vectors


def _already_complete(*, entry_count: int, source_csv_sha256: str) -> bool:
    """True iff the final `data/hs_taxonomy_embeddings.*` files already
    exist, are fully populated, and match the current taxonomy CSV — lets
    this script be safely invoked unattended on a recurring daily schedule
    (e.g. a `launchd`/`cron` job re-run automatically while a free-tier
    key's daily quota is being spread across multiple days) without
    re-embedding the entire corpus from scratch, at real API cost, every
    single day forever after the first successful completion."""
    meta_path = _OUTPUT_DIR / f"{_OUTPUT_STEM}.meta.json"
    npy_path = _OUTPUT_DIR / f"{_OUTPUT_STEM}.npy"
    if not (meta_path.exists() and npy_path.exists()):
        return False
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    matches_csv: bool = meta.get("source_csv_sha256") == source_csv_sha256
    matches_count: bool = meta.get("count") == entry_count
    return matches_csv and matches_count


async def main() -> None:
    settings = get_settings()
    entries = get_hs6_taxonomy_entries()
    source_csv_sha256 = _sha256_of(_TAXONOMY_CSV)

    if _already_complete(entry_count=len(entries), source_csv_sha256=source_csv_sha256):
        print(
            f"data/{_OUTPUT_STEM}.* is already up to date for the current taxonomy CSV "
            f"({len(entries)} entries) — nothing to do, no API calls made."
        )
        return

    print(f"Embedding {len(entries)} HS6 taxonomy descriptions via {settings.model_embedding}...")

    client = GeminiEmbeddingsClient(model=settings.model_embedding, api_key=settings.gemini_api_key)
    texts = [entry.description for entry in entries]
    already_embedded = _load_progress(source_csv_sha256=source_csv_sha256)
    vectors = await _embed_remaining(
        client, texts, already_embedded=already_embedded, source_csv_sha256=source_csv_sha256
    )
    if vectors is None:
        sys.exit(1)

    array = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    array = np.divide(array, norms, out=np.zeros_like(array), where=norms > 0)

    npy_path = _OUTPUT_DIR / f"{_OUTPUT_STEM}.npy"
    hscodes_path = _OUTPUT_DIR / f"{_OUTPUT_STEM}.hscodes.txt"
    meta_path = _OUTPUT_DIR / f"{_OUTPUT_STEM}.meta.json"

    np.save(npy_path, array)
    hscodes_path.write_text("\n".join(entry.hs_code for entry in entries) + "\n", encoding="utf-8")
    meta_path.write_text(
        json.dumps(
            {
                "model": settings.model_embedding,
                "task_type": _TASK_TYPE,
                "dims": int(array.shape[1]),
                "count": int(array.shape[0]),
                "source_csv_sha256": source_csv_sha256,
                "generated_at": datetime.now(UTC).isoformat(),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    _clear_progress()
    print(f"Wrote {npy_path} shape={array.shape}")
    print(f"Wrote {hscodes_path} ({len(entries)} lines)")
    print(f"Wrote {meta_path}")


if __name__ == "__main__":
    asyncio.run(main())
