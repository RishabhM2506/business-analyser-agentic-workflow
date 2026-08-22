"""Free-text product search: BM25 + vector embeddings + reciprocal rank
fusion + LLM rerank (2026-08-20 roadmap decision — see docs/PLAN.md).

Distinct from `app.knowledge.provider.KnowledgeProvider`, which is the
inverse operation (describe an already-known HS6 code) — this package
answers "what HS6 codes might this free text mean", the step that happens
*before* a code is selected. The existing analysis pipeline (`app.graph`)
is untouched; `app.main`'s new search endpoint hands off a selected code to
it exactly the way the existing item picker already does.
"""
