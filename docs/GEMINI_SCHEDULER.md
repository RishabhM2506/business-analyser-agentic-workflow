# Gemini Provider Scheduler

`app/gemini_scheduler/` — a quota-aware, health-aware, project-aware, concurrency-aware scheduler for
Gemini API calls, replacing the earlier `LoadBalancedGeminiModelClient`'s blind per-key round-robin
(`app/models.py`, removed 2026-09-04). Answers *"what capacity is safely available right now?"*, not
*"which key comes next?"*.

**Scope**: built in-memory, single-process — this repo runs one instance today (`docs/PLAN.md §1.2`), and
both existing rate-limiting mechanisms (`app/budget.py`'s `BudgetTracker`, `app/rate_limit.py`) already
follow that same pattern. Redis is provisioned in this repo but has no live application-code usage
(`app/fx/cache.py`'s `FxCache` takes an injected `RedisLike` client that nothing currently constructs), so
a distributed Redis-backed version of this scheduler was deliberately **not** built — see "Known
limitations" below for what that would take if the app ever goes multi-instance.

## Why this exists

The 7-key pool this app already had (`Settings.gemini_key_pool`) rotated keys blindly on any failure. That
helped against genuinely per-key problems (one key rate-limited, one key's connection hanging) but did
nothing for a real, live-observed failure mode this session: Gemini's own `503 "high demand"` on
`gemini-flash-latest`, which can hit every key against the same overloaded model near-simultaneously — no
amount of key rotation helps when the *model*, not the *key*, is the bottleneck. Worse, nothing distinguished
"this project's daily quota is gone, stop trying it" from "this project is briefly rate-limited, back off and
retry" — both surfaced identically as a rotate-to-next-key event.

## The credential → project relationship

Gemini quotas are primarily associated with the **Google Cloud project** behind a credential, not the API
key string itself. Two credentials belonging to the same project share one quota pool and must not be
treated as independent capacity — `app/gemini_scheduler/credentials.py`'s whole job is this grouping:

```
Google Account -> Google Cloud Project -> Gemini API Credential -> Gemini Model -> Quota/Health/Capacity
```

Every credential in the pool is a `GeminiCredential(id, project_id, api_key, account_label, models,
enabled)`. **This repo's actual current 7 keys have unknown project associations** (never independently
confirmed against the Google Cloud console) — the safe default assumes each key is its own separate
project (see "Configuration" below), never incorrectly consolidating them.

## Architecture

```
GeminiScheduler (app/gemini_scheduler/scheduler.py)   <- the ModelClient the app actually calls
    |
    +-- errors.py        classify_error(exc) -> GeminiErrorClass, retry_action_for(class) -> RetryAction
    +-- credentials.py   GeminiCredential pool, grouped by project_id
    +-- health.py         circuit breaker (per project+model), daily-quota tracking (per project),
    |                      credential health (per credential) - three independent scopes, never one
    |                      global boolean
    +-- concurrency.py    hierarchical AIMD: global limiter + one limiter per (project, model)
```

Full design rationale lives in each module's own docstring (this repo's convention — see e.g.
`app/models.py`'s own extensive header) rather than duplicated here; this document is the map, not the
territory.

## Routing algorithm (`GeminiScheduler._select_and_acquire`)

For each attempt:

1. **Filter** the credential pool to candidates that are: not already tried this call, model-compatible,
   credential-healthy (`health.is_credential_healthy`), and not daily-exhausted
   (`health.is_daily_exhausted`).
2. **Score** the survivors: `health_score * 2.0 + min(concurrency_headroom, 5) * 0.1`, with a rotating
   tiebreak (`health.HealthStore` + a shared `itertools.count`, the same fairness trick
   `LoadBalancedGeminiModelClient` established) so equally-scored candidates take turns rather than one
   always winning.
3. **Atomically claim** the top-scored candidate's circuit-breaker slot (`health.try_acquire` — a HALF_OPEN
   circuit only grants one concurrent trial) and its concurrency slot (`concurrency.try_acquire`), together.
   If concurrency is saturated after the circuit slot was claimed, the claim is released without recording
   any outcome (`health.cancel_acquire`) and the next-best candidate is tried.
4. **Dispatch** the real call. On success: record success, additive-increase that candidate's concurrency
   limit, release the slot. On failure: classify the error, record it against health, multiplicative-decrease
   concurrency if it was a real capacity-pressure signal (429/503), release the slot.

If nothing is eligible but a health-eligible candidate exists that's only blocked on concurrency, the
scheduler waits briefly (a short fixed poll, separate budget from dispatch attempts — see
`scheduler._DEFAULT_MAX_CAPACITY_WAIT_ATTEMPTS`) rather than failing immediately, since that saturation
resolves as soon as any in-flight call finishes.

**Retry policy** (`errors.retry_action_for`) is a 3-way split, not a boolean:

| Outcome | Examples | Behavior |
|---|---|---|
| `FAIL_FAST` | 400, 404, schema validation, safety block, cancellation | Re-raised immediately, no retry, no penalty |
| `RETRY_DIFFERENT_CANDIDATE` | 401, 403, daily quota (429) | Durable for *this* candidate — try a different one immediately, no backoff wait |
| `RETRY_WITH_BACKOFF` | 429 rate-limit, 500, 503, 504, 409-aborted, unknown | Transient — exponential backoff + jitter before the next attempt |

On total exhaustion (every attempted candidate failed, none remain), the scheduler **re-raises the real
last exception, never a wrapper** — `app/main.py`'s `isinstance(exc, ValidationError |
OutputParserException)` classification, and the two 2026-09-03 fixes in `app/nodes/summarize.py`/
`app/report/narrative.py` (broad `except Exception` around the model call), all depend on this.

## Health scoring & the circuit breaker

Purely **observed-signal-driven** — Google doesn't expose real per-project RPM/TPM/RPD ceilings
programmatically here, so there are no hard quota numbers to schedule against (see "Known limitations").
Per `(project_id, model)`:

- An EWMA of success/failure (`_EWMA_ALPHA = 0.2`) is the health score.
- `CLOSED -> OPEN` after `DEFAULT_CONSECUTIVE_FAILURE_THRESHOLD` (3) consecutive penalized failures, or
  immediately on a 403 (spec: "strong" project penalty).
- `OPEN` rejects everything until `cooldown_until` (exponential: `open_duration * 2^(open_count-1)`,
  capped at `DEFAULT_MAX_OPEN_DURATION_SECONDS`), then becomes `HALF_OPEN` and grants exactly one trial.
- Trial succeeds -> `CLOSED`, `open_count` resets. Trial fails -> `OPEN` again with an extended cooldown.
- Never permanently open.

**429 vs. daily quota**: both are Gemini status `RESOURCE_EXHAUSTED`; the only signal distinguishing them
is message text (`"per day"`/`"daily"`/`"PerDay"`). No real daily-exhaustion response was captured live
this session to confirm Gemini's exact wording, so this defaults to the safer `RATE_LIMITED`
interpretation when it can't tell (a few extra retries, self-correcting) rather than risk incorrectly
blackholing a project that could still serve requests. **This should be re-verified against a real daily
exhaustion response if one is ever captured.**

**Daily quota** (`health.is_daily_exhausted`, tracked per-project, not per-model — RPD is a project-wide
ceiling) excludes a project from routing for a conservative fixed cooldown from detection
(`DEFAULT_DAILY_QUOTA_COOLDOWN_SECONDS`, 24h) — not tied to a specific reset timezone, since Google's exact
per-project reset time isn't independently confirmed for these accounts either.

**401** disables only the specific credential (`DEFAULT_CREDENTIAL_COOLDOWN_SECONDS`, 1h) — a sibling
credential in the same project is unaffected, since an invalid/revoked key says nothing about the
project's own health.

## Concurrency

Hierarchical: every dispatch acquires a GLOBAL limiter, then a per-`(project, model)` limiter, both or
neither. AIMD (Additive Increase Multiplicative Decrease — chosen as the standard, well-understood
approach for adapting to observed capacity when the real ceiling is unknown): success -> `limit + 1`
(capped); a 429/503 -> `limit // 2` (floored). The global ceiling only shrinks (never AIMD-grows past its
configured value) and only when **at least `DEFAULT_DEGRADED_PROJECT_THRESHOLD` (3) distinct projects**
show congestion — a single project's own trouble never throttles everyone else (spec's own distinction:
"if the *majority* of projects experience 503, reduce global concurrency").

## Configuration

Extends `app/settings.py`. Two ways to configure credentials:

**Legacy (today's default, zero changes needed)**: `GEMINI_API_KEY` + `GEMINI_API_KEYS_EXTRA` (JSON array)
— each key becomes its own synthetic project (`legacy-0`, `legacy-1`, ...), matching the "never assume
consolidation" safety rule above.

**Structured, project-grouped** (`GEMINI_CREDENTIALS`, JSON array) — used once real project associations
are known:

```json
[
  {
    "id": "credential-01",
    "env_var": "GEMINI_CREDENTIAL_01",
    "project_id": "my-project-a",
    "account_label": "team-member-a@example.com",
    "enabled": true
  },
  {
    "id": "credential-02",
    "env_var": "GEMINI_CREDENTIAL_02",
    "project_id": "my-project-a",
    "account_label": "team-member-a@example.com"
  },
  {
    "id": "credential-11",
    "env_var": "GEMINI_CREDENTIAL_11",
    "project_id": "my-project-b",
    "account_label": "team-member-b@example.com",
    "models": ["gemini-flash-latest"]
  }
]
```

Each entry names an env var holding the real key — **the key itself never appears in this config**, so
it's always safe to log/inspect. `credential-01` and `credential-02` above share `project_id`
`my-project-a` and are correctly treated as one quota pool, not two.

### Adding another credential

Purely configuration, no source changes:

1. Set the new key in a fresh env var, e.g. `GEMINI_CREDENTIAL_11=<the real key>`.
2. Add an entry to `GEMINI_CREDENTIALS` naming that env var and the credential's real `project_id`.
3. Restart. `app.gemini_scheduler.credentials.build_credential_pool` fails loudly at startup if an
   *enabled* entry's env var is missing or blank — a disabled entry (`"enabled": false`) never requires
   its env var, so a credential can be pre-declared ahead of having the real secret.

## Operational troubleshooting

Every decision point logs a structured `gemini_scheduler.*` event (never a raw key — only
`credential_id`/`project_id`/`model`):

- `gemini_scheduler.dispatch_succeeded` / `attempt_failed` (with `error_class`) / `all_candidates_exhausted`
- `gemini_scheduler.circuit_opened` / `circuit_half_open` / `circuit_closed`
- `gemini_scheduler.daily_quota_exhausted` / `credential_disabled`

A project stuck excluded from routing: check `daily_quota_exhausted` (24h cooldown) vs. `circuit_opened`
(15s–300s cooldown, auto-recovers via `circuit_half_open`) in the logs to tell which scope is active.

## Known limitations

- **No real Google-advertised RPM/TPM/RPD numbers** are used anywhere — concurrency is purely
  observed-signal AIMD, not scheduled against a known ceiling with a safety factor. If real per-project
  quota numbers are ever obtained (Google Cloud Console), a token-aware admission layer using them would
  be a natural follow-on, not built now since fabricating numbers to schedule against would be worse than
  not guessing.
- **No token/TPM estimation** — requests are counted, not sized by estimated input tokens.
- **429-daily-vs-temporary classification is best-effort text matching**, not yet confirmed against a real
  daily-exhaustion response (see "Health scoring" above).
- **Single-process only.** If this app is ever deployed as multiple instances or behind a real async job
  queue, this scheduler's in-memory state would need to move to Redis (the spec's original design) —
  `health.py`'s three dict stores and `concurrency.py`'s counters are the pieces that would need atomic
  Redis-Lua equivalents; `errors.py`'s classification logic is stateless and would carry over unchanged.
- **No Postgres event history** — `gemini_scheduler.*` structured logs are the only record; nothing is
  durably persisted across a restart. Acceptable at today's scale; would need a `gemini_request_events`
  table (spec's own suggestion) if durable cross-restart history becomes valuable.
