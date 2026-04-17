# Design Notes

`liteagent` is a Python port of the core agent loop from [pi-mono](https://github.com/badlogic/pi-mono).

This file is intentionally not a full spec. The implementation already exists.
For exact behavior, use:

- the current code in `liteagent/`
- the local `pi-mono` source in `../pi-mono/packages/agent/src/`
- the tests in `tests/`

This document exists to capture the motivation, architecture, and the important
ways this project is similar to or different from `pi-mono`.

## What This Project Is

`liteagent` is a minimal, framework-agnostic agent core:

- raw loop functions for direct use
- a stateful `Agent` wrapper for normal app usage
- an event stream instead of a built-in transport
- a simple tool protocol

It is meant to plug into a CLI, web app, Slack bot, or anything else without
owning the UI, persistence, or application logic.

## Inspiration

The main inspiration is `pi-mono`'s `packages/agent`:

- dual loop architecture
- steering and follow-ups
- sequential tool execution
- a stateful `Agent` on top of a stateless loop
- event-driven streaming

The project is not trying to re-create the whole `pi` stack. It borrows the
core loop because that part is small, sharp, and proven.

## What We Keep From `pi-mono`

These behaviors are intentional carry-overs:

- inner loop for `LLM -> tool calls -> tool results -> LLM`
- outer loop for follow-up messages after the agent would otherwise stop
- steering as an interruption mechanism checked after each tool
- sequential tool execution so steering can skip remaining tools
- `Agent` wrapper with `prompt()`, `continue_run()`, `steer()`, `follow_up()`,
  `abort()`, and `wait_for_idle()`
- event-oriented design instead of binding to SSE, websockets, or a specific UI

If `liteagent` and `pi-mono` differ on one of those core behaviors, that should
usually be treated as a bug or an explicitly documented local difference.

## What We Deliberately Change

### LiteLLM instead of a custom provider layer

`pi-mono` has a large provider layer. `liteagent` uses `litellm` instead.

That trade buys:

- much less provider-specific code
- easier access to multiple model vendors
- fewer moving parts in this repo

It also costs us something:

- provider semantics still leak through even when response shapes are normalized
- thinking/reasoning metadata is handled inconsistently across providers (see
  [Known LiteLLM-Driven Differences](#known-litellm-driven-differences) below)
- some important metadata lives in LiteLLM-specific fields
- behavior can drift when LiteLLM changes upstream

### Python-native implementation choices

Compared to `pi-mono`, this repo uses:

- `asyncio.Event` instead of `AbortSignal`
- Pydantic instead of AJV for tool argument validation/coercion
- plain message dicts instead of TypeScript unions (construction is centralized
  via `_build_assistant_message` and `_build_tool_result_message` factories in
  `types.py`; a `TypedDict` layer for read-site type checking is a natural next
  step if the codebase grows)
- a single-consumer `EventStream` plus `Agent.subscribe()`
- unified `message_update` events with a `delta_type` field instead of pi's
  granular event types (`text_start`/`text_delta`/`text_end`,
  `thinking_start`/`thinking_delta`/`thinking_end`, etc.)

These are practical Python choices, not attempts to redesign the loop.

## Architecture

The core package is intentionally small:

- `stream.py`: async event stream
- `types.py`: shared vocabulary
- `loop.py`: stateless core loop
- `convert.py`: default `convert_to_llm` — the sole provider-specific boundary
- `agent.py`: stateful wrapper

The split matters:

- `loop.py` should stay focused on control flow
- `agent.py` should stay focused on state, queues, and subscribers
- `convert.py` is the only file that knows about provider differences
- transport, UI, storage, and app-specific tools belong outside the library

## Design Principles

- Keep the core minimal.
- Keep the loop provider-agnostic.
- Push app-specific behavior to the consumer.
- Prefer explicit control flow over framework magic.
- Preserve parity with `pi-mono` where that parity is the point of the project.

## Provider Boundary Principle

The loop and agent should not contain provider-specific branches like
`if anthropic` or `if gemini`.

Provider leakage is acceptable in two forms only:

- defensive reads at the LiteLLM boundary when equivalent data appears in
  different shapes
- opaque preservation of fields we do not interpret, such as
  `provider_specific_fields`, `thinking_blocks`, and `reasoning_content`

All provider-specific logic lives in `convert.py`. Currently this means:

- stripping liteagent metadata fields (denylist, not allowlist — new fields
  from litellm pass through automatically)
- hoisting images from tool results into user messages for OpenAI, which
  ignores image blocks in tool result content

If litellm is ever replaced, `convert.py` is the boundary to update.

## Tool Model

Tools are intentionally simple:

- a name, description, and JSON schema for the model
- an async `execute()` function for runtime behavior
- optional validation/coercion before execution
- optional streaming updates back to the consumer

The important behavioral point is that tools run one at a time. That is not a
performance oversight. It is how steering works.

## What This Library Does Not Own

`liteagent` does not try to own:

- your tools
- your system prompt
- your transport layer
- your UI
- your persistence/session model
- your app-specific compaction or memory strategy

The library provides the loop primitives. Consumers build the product.

## Known LiteLLM-Driven Differences

These are the main places where `liteagent` differs from `pi-mono` because it
leans on LiteLLM:

### Thinking metadata is not uniform

litellm promotes thinking to named fields, but inconsistently:

| Provider | Thinking text | Cryptographic signatures |
|---|---|---|
| **Anthropic** | `reasoning_content` (string) | `thinking_blocks` (first-class field) |
| **Gemini Pro** | `reasoning_content` (string) | `provider_specific_fields["thought_signatures"]` |
| **Gemini Flash** | `reasoning_content` — only on hard prompts, absent on trivial ones | same as Pro |
| **GPT-5.2** | **Not surfaced** via Chat Completions | n/a |

Both Anthropic and Gemini need signatures round-tripped for multi-turn thinking.
But Anthropic gets a dedicated `thinking_blocks` field while Gemini's equivalent
is buried in `provider_specific_fields`. This is why we preserve both fields on
every assistant message.

**Gemini `thought_signatures` nuance:** litellm injects a dummy signature
fallback (`skip_thought_signature_validator`) for tool/function-call replay in
`factory.py`, but for plain assistant text replay it just omits the field if
missing. Dropping signatures is non-fatal but weakens multi-turn fidelity.

**GPT-5.2 reasoning gap:** GPT-5.2 thinks internally (spends reasoning tokens)
but the thinking is hidden through `litellm.acompletion()`. OpenAI exposes
reasoning summaries only via the **Responses API** with
`reasoning={"effort": "high", "summary": "auto"}` and
`include=["reasoning.encrypted_content"]` for round-tripping. Pi uses the
Responses API for OpenAI reasoning models and gets full visibility. Our Chat
Completions path can't access this — a fix would require litellm adding
Responses API support or us calling the OpenAI API directly for GPT-5 models.

### Tool-result images are provider-sensitive

Anthropic and Gemini handle image-bearing tool results natively. OpenAI's Chat
Completions API ignores image blocks in tool result content (confirmed
empirically — the model receives the text but not the image).

The default converter in `convert.py` handles this: for OpenAI models, it hoists
images from tool results into synthetic user messages. This is transparent to
consumers — multimodal tool results work identically across all providers.

### Usage shapes vary

Cache token accounting is not exposed in one perfectly stable shape across all
providers. Anything around usage extraction needs to stay defensive.

**litellm cache token bug (as of March 2026):** litellm folds Anthropic's
`cache_read_input_tokens` into `prompt_tokens` and zeros out the cache-specific
fields. The slow test `test_anthropic_cache_tokens` is `xfail` until fixed
upstream.

```
Raw Anthropic SDK:  input=9   cache_read=1809  ← correct
litellm:            prompt=1818  cache_read=0   ← bug
```

### Streaming assembly is partly ours

LiteLLM gives normalized chunks, but not a full incremental event system with
the exact semantics this library wants. `liteagent` still owns the assembly of
streaming events that the rest of the library consumes.

**`stream_chunk_builder` finish_reason bug:** when `stream_options` includes
usage, providers send a final usage-only chunk with `finish_reason=None`.
`stream_chunk_builder` overwrites the real finish reason (e.g. `"length"`) with
that `None`, then defaults to `"stop"`. Our workaround: capture `finish_reason`
from chunks during streaming in `stream_llm_response()` instead of trusting the
builder's value.

## Model-Specific Notes

These are model-specific details not covered above. If they stop being true,
the code or tests should probably change too.

- **Anthropic** cache tokens are top-level on `usage`, not under
  `prompt_tokens_details`. See the litellm cache token bug above.
- **Gemini Flash** `reasoning_content` appears inconsistently on easier prompts.
  No visible reasoning text is not the same as no thinking support.
- **GPT-5.2** `reasoning_effort` passes through correctly; the gap is only in
  reasoning *output* visibility (see GPT-5.2 reasoning gap above).

## Pi-Mono Upstream Changes Not Ported

Audit window: ~Feb–Apr 2026. Tracked here so future ports can pick up
deliberately rather than rediscover. None are urgent; all are "consider when
a real use case appears."

**Most relevant to current consumers (`../agents`):** `Agent.signal` getter
and awaited subscribers. The first matters if external async work needs to
react to `agent.abort()`; the second matters the day anyone writes an
`async def` subscriber (today's sync subscribers will silently drop the
coroutine). The other six entries below are not relevant to anything in
`../agents` or its TODO.

### Parallel tool execution (`toolExecution: "parallel" | "sequential"`)
Pi runs tool calls in an assistant turn concurrently, with a sequential
preflight (validate args, run gating hooks) before fan-out. Liteagent is
sequential by design — keeps steering simple. Skip unless multi-tool latency
becomes a real complaint.

### `prepareArguments` tool hook (pi commit `b5f425ad`, 2026-03-29)
Optional per-tool function that runs *before* schema validation, used to fold
legacy argument shapes (e.g. old `oldText`/`newText`) into the current
schema. Only matters for resuming saved sessions across tool-schema changes.
Skip unless liteagent grows long-lived persisted sessions.

### `onPayload` / `onResponse` provider hooks (pi commits `a3f05423`, `d131fcd4`)
Per-LLM-call callbacks: inspect (and optionally rewrite) the outgoing
payload, inspect HTTP status + response headers (rate-limit headers
especially). Already achievable via litellm's own callback system; skip
unless a thin liteagent-level wrapper becomes worth the discoverability.

### `Agent.signal` getter (pi commit `7d4faa08`, 2026-03-28)
Exposes the active cancellation token to outside callers, so external async
work (background tasks, websocket pings, typing indicators) can react to
`agent.abort()`. Trivial to add (~5 lines) when a real use case appears.

### Awaited subscribed event handlers (pi commit `9022a5b5`, 2026-03-30)
**Breaking change in pi.** Subscribers were sync `(event) => void` (which is
what liteagent ports today); pi changed to `(event, signal) => Promise|void`
and the loop awaits each one before settling `prompt()`. Without this,
`async def` subscribers in liteagent silently fail (coroutine never
awaited). Add when any user writes an async subscriber and expects
`await agent.prompt()` to wait for it.

### `beforeToolCall` / `afterToolCall` hooks
Gate tool execution (`{block: true, reason}`) or rewrite tool results.
Liteagent's stance is "wrap your tools yourself." Reconsider if multiple
consumers reimplement audit/permission/redaction the same way.

### Deferred steering (pi commit `208a2cc1`, 2026-03-16)
Pi changed steering from "interrupt mid-batch, skip remaining tools" to
"let all tools finish, then process steer at turn boundary." Liteagent
intentionally keeps the interrupt model — feels more responsive, saves
tokens, emits explicit "Skipped due to queued user message" results
(marked `is_error=True`). Different design, not a bug. Don't port.

### AgentState property-access refactor (pi commit `cbe1a8b7`, 2026-03-30)
TS-specific copy-on-write via getter/setter on `state.tools` /
`state.messages`, replacing `setTools()` / `setMessages()` methods.
Liteagent's `set_*` methods are perfectly Pythonic. Don't port.

## Current Source-of-Truth Order

If these disagree, use this order:

1. current code in `liteagent/`
2. tests in `tests/`
3. `pi-mono` source for intended parity
4. this document for rationale and documented differences
