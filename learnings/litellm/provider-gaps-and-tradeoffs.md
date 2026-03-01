# Learning: litellm Provider Gaps and Tradeoffs

litellm normalizes the *shape* of LLM responses (everything becomes
`choices[0].delta.*`) but does not normalize the *semantics*. Downstream code
must still know where certain fields live per provider.

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

## What litellm normalizes well

These are safe to depend on across all providers:

- `delta.content` (text)
- `delta.reasoning_content` (thinking/reasoning)
- `delta.thinking_blocks` (Anthropic cryptographic signatures)
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
