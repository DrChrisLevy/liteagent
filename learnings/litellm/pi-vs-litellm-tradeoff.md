# Learning: Pi's Provider Layer vs Our litellm Shortcut

## How pi handles providers

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

## How we handle providers

We use litellm (0 lines of provider code). litellm normalizes the common
fields (content, tool_calls, finish_reason) but stuffs provider-specific
things into an untyped `provider_specific_fields` dict. We have to:

1. Know that `provider_specific_fields` exists
2. Preserve it through our message pipeline
3. Know where cache tokens live per provider (different locations)
4. Discover and fix gaps as they appear

## The tradeoff

| | Pi | Us (litellm) |
|---|---|---|
| Provider code | ~6,820 lines | 0 lines |
| Provider-specific bugs | Never — types are exhaustive | Periodic — litellm leaks |
| Adding a provider | ~500-900 lines per provider | Change one model string |
| Cache tokens | Normalized to `cacheRead`/`cacheWrite` | Different locations per provider |
| Thought signatures | `thoughtSignature` on ToolCall | `provider_specific_fields` bag |
| Maintenance | Update when APIs change | Update when litellm changes |

## Where this leaves us

For a learning project targeting 5 models, litellm is the right trade.
For production infrastructure, the periodic maintenance cost adds up —
every new provider field litellm stuffs into `provider_specific_fields`
is a potential silent data loss we need to discover and fix.

## Future option

If this becomes production quality, consider a thin provider layer for just
our 3 target providers (Anthropic, Google, OpenAI). Doesn't need to be
6,800 lines — pi supports ~20 providers with OAuth, transports, compat flags.
A focused layer for 3 providers could be much smaller, while giving us pi's
guarantees: typed fields, no escape hatches, no silent data loss.
