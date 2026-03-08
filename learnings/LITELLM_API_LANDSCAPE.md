# LiteLLM API Landscape: Chat Completions vs Responses API

*March 2026 — based on litellm source inspection, OpenAI docs, and empirical testing.*

This document captures the current state of litellm's two API paths, how
reasoning/thinking works across providers, and what it means for liteagent.

## The Two APIs

LiteLLM exposes two top-level interfaces:

- **`litellm.acompletion()`** — Chat Completions (`/v1/chat/completions`)
- **`litellm.aresponses()`** — Responses API (`/v1/responses`)

These are completely separate code paths with different provider configs,
different parameter shapes, and different streaming formats.

### The Bridges

LiteLLM has two bridge layers that convert between the APIs:

**Responses → Chat Completions** (`completion_extras/litellm_responses_transformation/`):
When a model is marked `"mode": "responses"` in the model cost map, calling
`acompletion()` silently redirects through this bridge:

1. Transforms chat messages into Responses API format
2. Calls `aresponses()` under the hood
3. Transforms the response back into Chat Completions shape

You can also force this by prefixing the model name with `responses/`
(e.g. `responses/gpt-5.4`).

**Chat Completions → Responses** (`responses/litellm_completion_transformation/`):
The reverse bridge — used when `aresponses()` is called for a model that only
supports Chat Completions. Transforms in the other direction.

### Which Models Use the Bridge?

Models marked `mode: responses` (as of March 2026):

- **Codex models**: `gpt-5-codex`, `gpt-5.1-codex`, `gpt-5.2-codex`, `gpt-5.3-codex`
- **Pro models**: `gpt-5-pro`, `gpt-5.2-pro`, `o3-pro`
- **Deep research**: `o3-deep-research`, `o4-mini-deep-research`
- **o1-pro**
- Various Azure/ChatGPT/Perplexity variants of the above

**Base chat models (`gpt-5`, `gpt-5.2`, `gpt-5.4`, all Anthropic, all Gemini)
are `mode: chat`** — they go straight to Chat Completions with no bridge.

## `litellm.modify_params`

liteagent sets `litellm.modify_params = True`. This enables structural fixes
to messages before they reach providers. Verified from source
(`litellm_core_utils/prompt_templates/factory.py`, `llms/anthropic/chat/transformation.py`):

**What it does:**

- **Orphaned tool calls**: Injects dummy tool result messages for tool calls
  that have no corresponding result
- **Orphaned tool results**: Removes tool result messages that reference
  non-existent tool call IDs
- **Empty content**: Replaces empty text content with placeholder text
- **Message alternation**: Inserts placeholder user messages for Anthropic/Bedrock
  when the first message isn't from the user
- **Thinking param**: Drops the `thinking` request parameter (NOT thinking blocks
  in messages) when no messages in the conversation have thinking blocks
- **Dummy tools**: Adds a dummy tool definition when messages contain tool calls
  but no `tools` parameter is provided

**What it does NOT do:**

- Does NOT strip/modify thinking_blocks, reasoning_content, or
  provider_specific_fields from messages
- Does NOT touch image content blocks in any message type
- Does NOT rewrite tool result content
- Does NOT modify assistant message content

`modify_params` is safe for liteagent — it only does structural housekeeping.

## Tool Result Images: The OpenAI Gap

**Verified empirically** (`investigate.py`) and **confirmed from OpenAI's type
definitions**: OpenAI's Chat Completions API does not support images in tool
result content. Tool messages accept `string` content only.

| Provider | Images in tool results | How |
|---|---|---|
| Anthropic | **Yes** | Native support — content blocks with `image_url` |
| Gemini | **Yes** | Native support — content blocks with `image_url` |
| OpenAI (Chat Completions) | **No** | Silently ignores `image_url` blocks |
| OpenAI (Responses API) | **Yes** | Bridge normalizes multimodal content |

**Empirical proof**: `investigate.py` sent `[text, image_url]` tool result content
to all providers. Anthropic and Gemini correctly identified image contents. OpenAI
(`gpt-5.2`, `gpt-5.4`) responded with generic/wrong answers, proving the image
was invisible. `gpt-5.3-codex` (which goes through the Responses API bridge)
correctly saw the image.

**liteagent workaround**: The default converter hoists images from tool results
into synthetic user messages for OpenAI models. This is transparent to the
consumer — multimodal tool results work identically across all providers.

## How `reasoning_effort` Is Translated Per Provider

liteagent passes `reasoning_effort` as a flat string to `litellm.acompletion()`.
LiteLLM translates this into each provider's native format:

### OpenAI (Chat Completions path)

Passed through as-is. It's a native OpenAI parameter for GPT-5+ and o-series.

### Anthropic

Mapped to Anthropic's `thinking` parameter
(`llms/anthropic/chat/transformation.py:_map_reasoning_effort`):

| reasoning_effort | Anthropic thinking param |
|---|---|
| `None` / `"none"` | Not sent (thinking disabled) |
| `"minimal"` | `{"type": "enabled", "budget_tokens": <MINIMAL>}` |
| `"low"` | `{"type": "enabled", "budget_tokens": <LOW>}` |
| `"medium"` | `{"type": "enabled", "budget_tokens": <MEDIUM>}` |
| `"high"` | `{"type": "enabled", "budget_tokens": <HIGH>}` |
| Any (Claude 4.6) | `{"type": "adaptive"}` — ignores the level |

Note: for Claude 4.6 models, **all** reasoning_effort values map to
`type: "adaptive"`, meaning Claude decides its own thinking budget.

### Gemini

Mapped to Gemini's `thinkingConfig`
(`llms/vertex_ai/gemini/vertex_and_google_ai_studio_gemini.py:_map_reasoning_effort_to_thinking_level`):

| reasoning_effort | Gemini thinkingConfig |
|---|---|
| `"minimal"` | `{"thinkingLevel": "minimal"}` (flash only, else `"low"`) |
| `"low"` | `{"thinkingLevel": "low", "includeThoughts": true}` |
| `"medium"` | `{"thinkingLevel": "medium"}` (3.1-pro/flash) or `"high"` (others) |
| `"high"` | `{"thinkingLevel": "high", "includeThoughts": true}` |
| `"disable"` / `"none"` | Lowest available level with `includeThoughts: false` |

Gemini 3 cannot fully disable thinking — the minimum is `"minimal"` for flash
models, `"low"` for others.

## How Thinking/Reasoning Content Comes Back

All three providers return reasoning through Chat Completions, but in different
shapes. LiteLLM partially normalizes this:

| Provider | `reasoning_content` | `thinking_blocks` | Signatures |
|---|---|---|---|
| Anthropic | String (promoted by litellm) | First-class field (with crypto signatures) | In `thinking_blocks` |
| Gemini | String (on hard prompts; absent on trivial ones) | Not used | `provider_specific_fields["thought_signatures"]` |
| OpenAI (GPT-5.x) | **Not surfaced** via Chat Completions | Not used | Not available |

**Anthropic duplication quirk:** litellm returns Anthropic thinking data in two
places simultaneously: `thinking_blocks` (top-level, with full thinking text +
signature) AND `provider_specific_fields["thinking_blocks"]` (with empty thinking
text, signature only). This is a litellm normalization artifact. Both fields
should be preserved on assistant messages for round-tripping; the top-level
`thinking_blocks` is the one Anthropic actually needs.

**Gemini thinking is optional per-turn:** Even with `reasoning_effort: "high"`,
Gemini may or may not include `reasoning_content` on a given turn. The
`thought_signatures` in `provider_specific_fields` are more consistently present.
Both must be round-tripped for multi-turn thinking fidelity.

**OpenAI reasoning gap confirmed:** `investigate.py` output shows GPT-5.2
spending `reasoning_tokens: 375` (visible in `usage.completion_tokens_details`)
while `reasoning_content`, `thinking_blocks`, and `provider_specific_fields` are
all empty. The model thinks but the thinking is invisible through Chat
Completions. OpenAI only exposes reasoning summaries via the Responses API with
`reasoning={"effort": "high", "summary": "auto"}`.

## GPT-5.4: The Reasoning + Tools Problem

GPT-5.4 has a specific limitation in the Chat Completions API: **reasoning and
tool calling cannot be used together.** LiteLLM works around this by silently
dropping `reasoning_effort` when tools are present
(`llms/openai/chat/gpt_5_transformation.py:188-197`):

```python
# gpt-5.4: function calls not supported when reasoning_effort != "none"
if self.is_model_gpt_5_4_model(model):
    has_tools = bool(non_default_params.get("tools") or optional_params.get("tools"))
    if has_tools and reasoning_effort not in (None, "none"):
        non_default_params.pop("reasoning_effort", None)
        optional_params.pop("reasoning_effort", None)
```

This is **only GPT-5.4** — earlier models (5.0, 5.1, 5.2, 5.3-codex) support
reasoning + tools together in Chat Completions.

The Responses API does not have this limitation. GPT-5.4 supports reasoning +
tools + "preambles" (brief explanations before tool calls) through the Responses
API. OpenAI's docs explicitly recommend the Responses API for reasoning models:

> "Reasoning models work better with the Responses API. While the Chat
> Completions API is still supported, you'll get improved model intelligence
> and performance by using Responses."

## The Trend

OpenAI is increasingly investing in the Responses API for agent workflows:

- GPT-5.0: reasoning + tools works fine in Chat Completions
- GPT-5.1/5.2: still works, added `none` and `xhigh` effort levels
- GPT-5.4: reasoning + tools **broken in Chat Completions**, works in Responses API
- Codex/Pro variants are **Responses-only** by default in litellm
- Responses API gets features Chat Completions doesn't (preambles, CoT caching,
  reasoning summaries)

Meanwhile, Anthropic and Gemini have no Responses API equivalent — their thinking
features work exclusively through Chat Completions (or their own native APIs,
which litellm wraps as Chat Completions).

## Responses API Provider Support in LiteLLM

`litellm.aresponses()` has uneven provider coverage:

| Provider | Responses API support |
|---|---|
| OpenAI | Full, native |
| Azure OpenAI | Full, native |
| Anthropic | **Experimental** — `llms/anthropic/experimental_pass_through/responses_adapters/` translates between Anthropic messages and Responses API shape |
| Gemini | **None** — no adapter exists |
| Perplexity, XAI, etc. | Partial, varies |

## What This Means for liteagent

liteagent currently uses `litellm.acompletion()` exclusively. This is the right
call for provider-agnostic operation — it's the only path that works uniformly
across Anthropic, Gemini, and OpenAI.

The cost:

1. **GPT-5.4 reasoning is silently disabled** when tools are present (which is
   always, in an agent loop). No error, just degraded intelligence.
2. **OpenAI reasoning content is invisible** — GPT-5.x spends tokens thinking
   but we can't see or round-trip the thoughts.
3. **OpenAI Responses API features are inaccessible** — preambles, CoT caching,
   reasoning summaries.

The default converter handles the multimodal gap (tool result images for OpenAI)
transparently. The remaining gaps are all reasoning-related and affect OpenAI only.

Possible future approaches:

- **Do nothing.** Accept the GPT-5.4 limitation. Use 5.2/5.3-codex for OpenAI
  reasoning + tools. This is the simplest path.
- **Use `responses/` prefix.** Force GPT-5.4 through the bridge to get reasoning
  + tools. Requires testing whether the bridge handles streaming, tool calls, and
  multi-turn correctly for this model.
- **Call `litellm.aresponses()` for OpenAI models.** Add a provider-specific code
  path. Violates the "no provider branches in the loop" principle but would give
  full OpenAI capabilities.
- **Wait for litellm.** If litellm adds transparent Responses API routing for
  GPT-5.4 (setting `mode: responses` in the cost map), this fixes itself without
  any liteagent changes.

## Sources

- litellm source: `../litellm/`
  - `litellm/llms/openai/chat/gpt_5_transformation.py` — GPT-5.x Chat Completions handling
  - `litellm/llms/anthropic/chat/transformation.py` — Anthropic reasoning_effort mapping
  - `litellm/llms/vertex_ai/gemini/vertex_and_google_ai_studio_gemini.py` — Gemini reasoning mapping
  - `litellm/responses/main.py` — Responses API entry point
  - `litellm/completion_extras/litellm_responses_transformation/` — Responses → Chat Completions bridge
  - `litellm/responses/litellm_completion_transformation/` — Chat Completions → Responses bridge
  - `litellm/litellm_core_utils/prompt_templates/factory.py` — `modify_params` sanitization logic
  - `litellm/model_prices_and_context_window_backup.json` — model mode/capability map
- OpenAI docs:
  - [Using GPT-5.4](https://developers.openai.com/api/docs/guides/latest-model/)
  - [GPT-5.4 Model](https://developers.openai.com/api/docs/models/gpt-5.4)
  - [Reasoning models](https://developers.openai.com/api/docs/guides/reasoning)
- Empirical testing: `investigate.py` — 8 models, 4-turn multimodal scenario
