# Framework Comparisons

How py-pi-agent compares to existing agent SDKs.

---

## py-pi-agent vs OpenAI Agents SDK

[OpenAI Agents SDK](https://openai.github.io/openai-agents-python/) (~19k GitHub stars) is the main
Python agent framework to compare against. Here's how we differ:

### What OpenAI SDK has that we don't (and whether we care)

| OpenAI SDK feature | Our stance |
|---|---|
| `@function_tool` decorator (auto JSON Schema from type hints) | Want this — Phase 4 TODO |
| Handoffs (agent-to-agent delegation) | Don't need — sub-agents via tools works fine |
| Guardrails (input/output validation) | Nice to have later — consumer can validate in hooks for now |
| Built-in tracing (sends to OpenAI dashboard) | Don't need — consumer's problem |
| Sessions (SQLite, Redis, SQLAlchemy) | Don't need — consumer handles persistence |
| Structured output (`output_type` → Pydantic model) | Could add later |
| MCP support | Could add later |
| Hosted tools (web search, code interpreter) | OpenAI-only, not relevant |
| `RunContextWrapper` (dependency injection) | Clean pattern — consider for Phase 4 |

### What we have that OpenAI SDK doesn't

| py-pi-agent feature | Why it matters |
|---|---|
| **Steering (mid-run interruption)** | User can redirect agent while it's executing tools. OpenAI SDK has no equivalent. |
| **Follow-up messages** | Queue messages that wait until agent finishes. OpenAI SDK doesn't support this. |
| **Streaming tool output (`on_update`)** | Tools can push partial results (e.g., bash output line-by-line). OpenAI SDK tools return all-or-nothing. |
| **`transform_context` hook** | Modify messages before each LLM call (compaction, injection, pruning). No equivalent in OpenAI SDK. |
| **Truly LLM-agnostic** | litellm-first, no vendor coupling. OpenAI SDK is OpenAI-first with real lock-in concerns. |
| **Minimal core** | ~300 lines of loop code. OpenAI SDK is a much larger surface area to understand. |

### Different philosophies

- **OpenAI SDK**: "Here's everything, just use OpenAI" — batteries-included product, optimized for OpenAI models, lots of built-in features, vendor lock-in risk.
- **py-pi-agent**: "Here's the core loop, bring your own everything" — learning project that could become a library, minimal, truly agnostic, pi's battle-tested patterns.

### Ideas to steal later

- `@function_tool` decorator for DX — auto-generate JSON Schema from Python type hints + docstrings
- Guardrails concept — validate at agent boundaries, tripwire mechanism to halt on bad output
- Structured output — let consumer specify a Pydantic model, agent must return matching JSON
- `RunContextWrapper` — clean dependency injection for passing app state into tools without globals

---

## py-pi-agent vs Anthropic Claude Agent SDK

[Claude Agent SDK](https://platform.claude.com/docs/en/agent-sdk/overview) (~4.9k GitHub stars) is
Anthropic's agent library — essentially Claude Code exposed as a programmable Python/TypeScript package.

### Fundamental difference

Claude Agent SDK is a **black box wrapper around Claude Code**. You don't implement tools or the loop.
You call `query()`, Claude executes built-in tools (Read, Edit, Bash, etc.) internally, and you get
back an async iterator of messages. py-pi-agent is the opposite — the loop is fully explicit and
you implement everything.

### What Claude SDK has that we don't (and whether we care)

| Claude SDK feature | Our stance |
|---|---|
| Built-in tools (Read, Edit, Bash, Grep, Glob, etc.) | Don't need — consumer implements tools |
| Lifecycle hooks (PreToolUse, PostToolUse, Stop, etc.) | Interesting — consider for Phase 4 |
| Permission system (allow/deny/ask per tool) | Consumer's problem |
| Subagents with per-agent model override | We support via tools naturally |
| Session resume + fork | Consumer handles persistence |
| Structured output (JSON Schema) | Could add later |
| MCP support (stdio, HTTP, SSE, in-process) | Could add later |
| `maxBudgetUsd` (cost limit) | Consumer can implement via transform_context or event subscription |
| File checkpointing + rewind | Specific to coding agents, not core |
| Sandbox configuration | Consumer's problem |

### What we have that Claude SDK doesn't

| py-pi-agent feature | Why it matters |
|---|---|
| **Explicit agent loop** | You understand and control every step. Claude SDK is opaque. |
| **Any LLM** | GPT, Gemini, DeepSeek, local models. Claude SDK is Claude-only. |
| **Custom tools as functions** | Define tools as async Python functions. Claude SDK requires MCP servers for custom tools. |
| **Streaming tool output (`on_update`)** | Partial results during execution. Claude SDK tools are all-or-nothing. |
| **`transform_context` hook** | Modify messages before each LLM call. No equivalent in Claude SDK. |
| **Granular event stream** | Token-level deltas, tool execution lifecycle. Claude SDK yields high-level messages. |
| **Learning opportunity** | You built it, you understand it. Claude SDK is a black box. |

### Ideas to steal from Claude SDK

- `maxBudgetUsd` — consumer can implement by tracking usage events and calling abort()
- Lifecycle hooks pattern (PreToolUse, PostToolUse) — cleaner than ad-hoc callbacks
- Session fork — branch conversations without losing the original

---

## Three-Way Philosophy Comparison

| | Claude Agent SDK | OpenAI Agents SDK | py-pi-agent |
|---|---|---|---|
| **Approach** | Claude Code as a library | Multi-agent framework | Core agent loop library |
| **Agent loop** | Black box | Visible but managed | Fully explicit |
| **Tools** | Built-in | Decorators + hosted | You define everything |
| **Models** | Claude only | OpenAI-first | Any (litellm) |
| **Lock-in** | Claude models | OpenAI ecosystem | None |
| **Target user** | "I want Claude Code in my app" | "I want a multi-agent system fast" | "I want to understand and control the loop" |
| **Complexity** | Medium | High | Low |
