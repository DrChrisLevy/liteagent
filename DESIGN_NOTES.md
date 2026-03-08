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

## Current Source-of-Truth Order

If these disagree, use this order:

1. current code in `liteagent/`
2. tests in `tests/`
3. `pi-mono` source for intended parity
4. this document for rationale and documented differences

## Related Docs

- `learnings/COMPARISONS.md` — how liteagent compares to OpenAI Agents SDK, Claude Agent SDK, and Pydantic AI
