# Code Review Tracker: liteagent

This file tracks review findings against two baselines:

- `../pi-mono/packages/agent/src/` for intended behavior parity
- `DESIGN_NOTES.md` for documented local differences and intended boundaries

The point of this file is triage, not prose. If a behavior is already documented in
the spec, mark it as a `sharp-edge` or `backlog` item instead of pretending it is
an undiscovered bug.

## Legend

- `open`: likely worth fixing now
- `sharp-edge`: current behavior is documented, but the default is still risky or misleading
- `backlog`: real design debt, not a correctness failure

Priority:

- `P1`: correctness or cancellation gap
- `P2`: target-model fidelity or user-visible behavior gap
- `P3`: maintainability / DX

## Active Summary

| ID | Priority | Status | Summary |
|---|---|---|---|
| `CR-001` | `P1` | `open` | Abort cannot interrupt a pending LLM startup await |
| `CR-002` | `P2` | `open` | Default Agent converter drops Gemini `thought_signatures` |
| `CR-003` | `P2` | `sharp-edge` | Default Agent converter drops tool-result images |
| `CR-004` | `P2` | `open` | Usage extraction misses nested `cache_creation_tokens` |
| `CR-005` | `P3` | `backlog` | Messages are still untyped dicts throughout the core loop |
| `CR-006` | `P3` | `backlog` | Agent and loop both append messages, relying on copied lists |

## Active Findings

### `CR-001` Abort cannot interrupt pending LLM startup

- Priority: `P1`
- Status: `open`
- Type: parity gap vs `pi-mono`
- liteagent:
  - `liteagent/loop.py:294-299`
- pi-mono baseline:
  - `../pi-mono/packages/agent/src/agent-loop.ts:233-237`
- Design notes:
  - `DESIGN_NOTES.md` sections `What We Deliberately Change` and `Provider Boundary Principle`
- Why this matters:
  - `agent.abort()` only flips an `asyncio.Event`.
  - `stream_llm_response()` does not consult that signal until after `await litellm.acompletion(**kwargs)` returns.
  - If provider startup is slow, rate-limited, or hung, the run stays busy even though the user already aborted.
- Why `pi-mono` is different:
  - `pi-mono` passes an `AbortSignal` into the provider stream function itself, so the request can be interrupted while startup is still pending.
- Suggested fix:
  - Make the LLM startup await cancellable, either by racing `acompletion()` against the signal or by wrapping the provider call in a task that can be cancelled safely.
- Tests to add:
  - pending `acompletion()` + pre-set signal => immediate `aborted`
  - pending `acompletion()` + mid-await `abort()` => run resolves without waiting for first chunk

### `CR-002` Default Agent converter drops Gemini `thought_signatures`

- Priority: `P2`
- Status: `open`
- Type: fidelity gap with a doc conflict
- liteagent:
  - `liteagent/agent.py:45-55`
- pi-mono baseline:
  - `../pi-mono/packages/agent/src/agent.ts:31-33`
- Design notes:
  - `DESIGN_NOTES.md` sections `Known LiteLLM-Driven Differences` and `Provider Boundary Principle`
- Why this matters:
  - The default converter preserves `thinking_blocks` and `reasoning_content`, but strips `provider_specific_fields`.
  - Gemini stores `thought_signatures` in `provider_specific_fields`.
  - The raw loop preserves them, but the default `Agent` path loses them unless the caller overrides `convert_to_llm`.
- Why this is not just a nit:
  - The provider notes explicitly say those signatures matter for multi-turn thinking fidelity.
  - The spec's default-converter section documents the current stripping behavior, but the provider notes point the other way. That is a doc conflict worth resolving, not just a taste issue.
- Suggested fix:
  - Preserve `provider_specific_fields` on assistant messages in the default converter, or ship a provider-aware default converter.
- Tests to add:
  - default Agent converter round-trip preserves Gemini-style `provider_specific_fields`

### `CR-003` Default Agent converter drops tool-result images

- Priority: `P2`
- Status: `sharp-edge`
- Type: documented difference that weakens the default API
- liteagent:
  - `liteagent/agent.py:58-68`
- pi-mono baseline:
  - `../pi-mono/packages/agent/src/agent.ts:31-33`
- Design notes:
  - `DESIGN_NOTES.md` section `Known LiteLLM-Driven Differences`
- Why this matters:
  - Tool results with mixed `text` + `image_url` content are flattened to text in the default converter.
  - Anthropic and Gemini can consume tool-result images, so multimodal tools lose information on the default Agent path.
- Why this is `sharp-edge`, not `open`:
  - The spec already documents the default converter as minimal and says consumers can override it for provider-specific multimodal behavior.
  - So this is not hidden behavior. It is still a bad default for a library that otherwise supports multimodal tool results.
- Suggested fix:
  - Either preserve `image_url` for providers that support it, or make the default converter provider-aware and apply the OpenAI workaround automatically.
- Tests to add:
  - default converter image behavior for Anthropic / Gemini expectations
  - default converter OpenAI fallback behavior if provider-aware conversion is added

### `CR-004` Usage extraction misses nested `cache_creation_tokens`

- Priority: `P2`
- Status: `open`
- Type: accounting bug at the LiteLLM boundary
- liteagent:
  - `liteagent/loop.py:53-61`
- Design notes:
  - `DESIGN_NOTES.md` section `Known LiteLLM-Driven Differences`
- Why this matters:
  - `_extract_usage()` reads top-level `cache_creation_input_tokens`, but not nested `usage.prompt_tokens_details.cache_creation_tokens`.
  - When LiteLLM uses the nested shape, cache-write accounting is under-reported even though the data exists.
- pi-mono note:
  - This is not a direct `pi-mono` parity issue. `pi-mono` normalizes usage upstream into a stable shape, so this specific boundary bug does not exist there.
- Suggested fix:
  - Fall back to `usage.prompt_tokens_details.cache_creation_tokens` when the top-level field is absent.
- Tests to add:
  - helper coverage for nested `cache_creation_tokens`

## Backlog

### `CR-005` Messages are still untyped dicts

- Priority: `P3`
- Status: `backlog`
- Type: maintainability debt vs `pi-mono`
- liteagent:
  - assistant messages are hand-built in multiple places in `liteagent/loop.py` and `liteagent/agent.py`
- pi-mono baseline:
  - typed message unions in `../pi-mono/packages/agent/src/types.ts`
- Why this matters:
  - The code repeatedly constructs near-identical assistant-message dicts with `role`, `content`, `tool_calls`, `thinking_blocks`, `reasoning_content`, `usage`, `stop_reason`, and `timestamp`.
  - A typo or omitted field becomes a silent `dict.get()` failure instead of a type error.
- Suggested fix:
  - Add a shared assistant-message factory first.
  - Consider a lightweight typed message layer after that if the codebase keeps growing.

### `CR-006` Message accumulation is coupled across Agent and loop

- Priority: `P3`
- Status: `backlog`
- Type: design risk
- liteagent:
  - `liteagent/agent.py:328-331`
  - `liteagent/agent.py:407-409`
  - `liteagent/loop.py:497`
  - `liteagent/loop.py:560-561`
- Design notes:
  - `DESIGN_NOTES.md` section `Architecture`
- Why this matters:
  - The loop appends to its local context, and the Agent also appends on every `message_end`.
  - This is safe today because the Agent snapshots message history into a copied list before the run starts.
  - If that copy disappears later, message accumulation will become subtly wrong.
- Why this is not an active bug:
  - The copy is intentional and documented in the spec.
  - So current behavior is correct; the problem is that the ownership boundary is easy to break during refactors.
- Suggested fix:
  - Make ownership explicit in code comments or consolidate message accumulation behind one layer.

## Low-Value Cleanup

These are real but not worth mixing into the active bug list:

- `_now_ms()` is duplicated in `liteagent/loop.py` and `liteagent/agent.py`
- `_dequeue_steering()` / `_dequeue_follow_ups()` use `list.pop(0)` instead of `deque`
- dataclass `__repr__` output is noisy for debugging
- the test mock helpers are duplicated between `tests/test_loop.py` and `tests/test_agent.py`

## Focused Test Gaps

If adding tests next, do these first:

- pending `litellm.acompletion()` cancellation before first chunk
- default Agent converter preservation of Gemini `provider_specific_fields`
- nested `prompt_tokens_details.cache_creation_tokens` usage extraction
- default Agent converter behavior for multimodal tool results
