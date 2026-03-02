# Learning: litellm Provider Gaps and Tradeoffs

litellm normalizes the *shape* of LLM responses (everything becomes
`choices[0].delta.*`) but does not normalize the *semantics*. Downstream code
must still know where certain fields live per provider.

## Provider boundary principle

`loop.py` is provider-agnostic but boundary-defensive:

- **No provider-specific control flow** — no `if anthropic:` or `if gemini:`
- **Defensive reads at the litellm boundary** — check multiple locations with
  `getattr` fallbacks (e.g., `_extract_usage`)
- **Opaque preservation** — pass through `provider_specific_fields`,
  `thinking_blocks`, etc. without interpreting them
- **Request-shape differences go in `convert_to_llm`** — that's the caller's job

If you're interpreting provider-specific semantics in loop.py, that's the smell.
See also: SPEC.md "Provider boundary principle."

## Cache tokens (fixed)

litellm preserves two different locations for cache token counts:

| Provider | Where cache tokens live |
|---|---|
| **OpenAI** | `usage.prompt_tokens_details.cached_tokens` (read only, no creation concept) |
| **Anthropic** | `usage.cache_creation_input_tokens` and `usage.cache_read_input_tokens` (top-level) |
| **Gemini** | No prompt caching in the API |

Our `_extract_usage()` checks both locations.

### litellm bug (as of March 2026)

litellm folds Anthropic's `cache_read_input_tokens` into `prompt_tokens` and
zeros out the cache-specific fields. Raw Anthropic SDK returns them correctly.
The slow test `test_anthropic_cache_tokens` is marked `xfail` until this is
fixed upstream.

```
Raw Anthropic SDK:  input=9   cache_read=1809  ← correct
litellm:            prompt=1818  cache_read=0   ← bug
```

## provider_specific_fields (fixed)

`stream_chunk_builder()` preserves `provider_specific_fields` on both the
message and individual tool calls. We now preserve them too at both levels.
Verified by `test_gemini_provider_specific_fields_preserved`.

### What's in provider_specific_fields

| Provider | What's in `provider_specific_fields` |
|---|---|
| **Gemini** (`gemini-3-flash-preview`, `gemini-3.1-pro-preview`) | `thought_signatures` — opaque tokens for multi-turn context preservation |
| **Anthropic** | Citations, web search results (on responses only). Anthropic does NOT read these from previous messages in conversation history. |
| **OpenAI** | Not used |

### Gemini thought_signatures — nuance

The dummy signature fallback (`skip_thought_signature_validator`) is narrower
than it sounds. litellm injects it for tool/function-call replay in
`factory.py`, but for plain assistant text replay it just omits
`thoughtSignature` if `provider_specific_fields["thought_signatures"]` is
missing in `transformation.py`. Dropping them would be non-fatal but would
lose fidelity in multi-turn thinking conversations.

### Still not preserved

Other litellm message fields we don't carry: `annotations`, `audio`, `images`.
None of our target models use these today.

## stream_options

We always send `"stream_options": {"include_usage": True}`. This is an
OpenAI-specific parameter. litellm strips or translates it for other providers.
Verified working for all 5 target models via `test_usage_present`.

## Streaming assembly

litellm has no incremental stream assembler. `stream_chunk_builder()` is
post-hoc only — it operates after all chunks are collected. Our 250-line
`stream_llm_response()` does what litellm's `ChunkProcessor` does, but
incrementally, for live `message_update` events. The plumbing is correct
(verified against `streaming_chunk_builder_utils.py`).

## Thinking field inconsistency

litellm promotes thinking to named fields — but inconsistently:

| Provider | Thinking text | Cryptographic signatures |
|---|---|---|
| **Anthropic** | `reasoning_content` (string) ✅ | `thinking_blocks` (first-class field with signatures) |
| **Gemini Pro** | `reasoning_content` (string) ✅ | `provider_specific_fields["thought_signatures"]` (opaque bag) |
| **Gemini Flash** | `reasoning_content` — only on hard questions, not trivial ones | same as Pro |
| **GPT-5.2** | **Not surfaced** via Chat Completions API | n/a |

Both Anthropic and Gemini need their signatures round-tripped for multi-turn thinking
conversations. But Anthropic gets a dedicated `thinking_blocks` field while Gemini's
equivalent is buried in `provider_specific_fields`. Same concept, different treatment.

This is why we preserve both `thinking_blocks` AND `provider_specific_fields` on every
assistant message — can't rely on litellm being consistent about where signatures land.

### GPT-5.2 reasoning gap (as of March 2026)

GPT-5.2 is a thinking model but its reasoning text is **not visible** through
`litellm.acompletion()` (Chat Completions API). Verified:

- `reasoning_effort="high"` is passed correctly (litellm's `gpt_5_transformation.py`
  lists it as a supported param)
- GPT-5.2 thinks internally (spends reasoning tokens)
- But the thinking is hidden — `reasoning_content` is always `None`

OpenAI exposes reasoning summaries via `reasoning.summary = "auto"` in the
**Responses API**, not Chat Completions. litellm's Chat Completions path doesn't
set this parameter, so reasoning text is never returned.

To get GPT-5.2 thinking output, you'd need litellm's Responses API path or a
direct OpenAI SDK call with `reasoning={"effort": "high", "summary": "auto"}`.

### How pi solves this

Pi uses the **Responses API** (not Chat Completions) for all OpenAI reasoning models.
It sends three things we don't:

```python
reasoning={"effort": "medium", "summary": "auto"}
include=["reasoning.encrypted_content"]
```

- `summary: "auto"` enables reasoning summary text in the response
- `include: ["reasoning.encrypted_content"]` enables round-tripping the encrypted
  reasoning back to OpenAI on subsequent turns (like Anthropic's `thinking_blocks`
  signatures)

Pi stores the full `ResponseReasoningItem` JSON as `thinkingSignature` on the
thinking block — same pattern as Anthropic/Gemini signatures, just different data.

Our `litellm.acompletion()` path uses Chat Completions which doesn't support any
of this. A fix would require either litellm adding Responses API support to
`acompletion`, or us calling the OpenAI Responses API directly for GPT-5 models.

## What litellm normalizes well

These are safe to depend on across all providers:

- `delta.content` (text)
- `delta.reasoning_content` (thinking/reasoning — Anthropic and Gemini only, not GPT-5.2 via Chat Completions)
- `delta.thinking_blocks` (Anthropic cryptographic signatures — not emitted for other providers)
- `delta.tool_calls` with `.index`, `.id`, `.function.name`, `.function.arguments`
- `role: "system"` → native system prompt format
- OpenAI tool schemas → native tool format
- `finish_reason` mapping (`end_turn`→`stop`, `tool_use`→`tool_calls`, etc.)
- `modify_params = True` auto-fixes (alternating roles, orphaned tool calls, etc.)

## How pi handles this differently

Pi has ~6,820 lines of hand-rolled provider code in `packages/ai/src/providers/`.
Each provider (Anthropic 880 lines, Google 940, OpenAI 824, Bedrock 739, etc.)
maps raw API responses into pi's unified message types.

The key insight: **pi promotes every provider-specific field into a typed,
first-class field on the unified types.** There is no `provider_specific_fields`
escape hatch.

```typescript
// Pi's ToolCall — thoughtSignature is a first-class field
interface ToolCall {
    type: "toolCall";
    id: string;
    name: string;
    arguments: Record<string, any>;
    thoughtSignature?: string;  // Gemini-specific, always here
}

// Pi's ThinkingContent — signatures are first-class
interface ThinkingContent {
    type: "thinking";
    thinking: string;
    thinkingSignature?: string;  // Anthropic/OpenAI
    redacted?: boolean;          // Anthropic redacted thinking
}

// Pi's Usage — cache tokens are normalized, same fields for all providers
interface Usage {
    input: number;
    output: number;
    cacheRead: number;    // every provider normalizes to this
    cacheWrite: number;   // every provider normalizes to this
    totalTokens: number;
    cost: { input, output, cacheRead, cacheWrite, total };
}
```

Because the types carry everything, `agent-loop.ts` (417 lines) never worries
about dropping provider metadata. `convertToLlm` knows exactly where to find
each field.

## The tradeoff

| | Pi | Us (litellm) |
|---|---|---|
| Provider code | ~6,820 lines | 0 lines |
| Provider-specific bugs | Never — types are exhaustive | Periodic — litellm leaks |
| Adding a provider | ~500-900 lines per provider | Change one model string |
| Cache tokens | Normalized to `cacheRead`/`cacheWrite` | Different locations per provider |
| Thought signatures | `thoughtSignature` on ToolCall | `provider_specific_fields` bag |
| Maintenance | Update when APIs change | Update when litellm changes |

For production quality, the periodic maintenance cost adds up — every new
provider field litellm stuffs into `provider_specific_fields` is a potential
silent data loss we need to discover and fix.

## Future option

Consider a thin provider layer for just our 3 target providers (Anthropic,
Google, OpenAI). Doesn't need to be 6,800 lines — pi supports ~20 providers
with OAuth, transports, compat flags. A focused layer for 3 providers could
be much smaller, while giving us pi's guarantees: typed fields, no escape
hatches, no silent data loss.
