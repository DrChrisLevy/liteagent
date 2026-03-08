# Framework Comparisons

How liteagent compares to existing agent SDKs.

---

## liteagent vs OpenAI Agents SDK

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

| liteagent feature | Why it matters |
|---|---|
| **Steering (mid-run interruption)** | User can redirect agent while it's executing tools. OpenAI SDK has no equivalent. |
| **Follow-up messages** | Queue messages that wait until agent finishes. OpenAI SDK doesn't support this. |
| **Streaming tool output (`on_update`)** | Tools can push partial results (e.g., bash output line-by-line). OpenAI SDK tools return all-or-nothing. |
| **`transform_context` hook** | Modify messages before each LLM call (compaction, injection, pruning). No equivalent in OpenAI SDK. |
| **Truly LLM-agnostic** | litellm-first, no vendor coupling. OpenAI SDK is OpenAI-first with real lock-in concerns. |
| **Minimal core** | Small, hackable codebase (5 files). OpenAI SDK is a much larger surface area. |

### Different philosophies

- **OpenAI SDK**: "Here's everything, just use OpenAI" — batteries-included product, optimized for OpenAI models, lots of built-in features, vendor lock-in risk.
- **liteagent**: "Here's the core loop, bring your own everything" — minimal, truly agnostic, pi's battle-tested patterns.

### Ideas to steal later

- `@function_tool` decorator for DX — auto-generate JSON Schema from Python type hints + docstrings
- Guardrails concept — validate at agent boundaries, tripwire mechanism to halt on bad output
- Structured output — let consumer specify a Pydantic model, agent must return matching JSON
- `RunContextWrapper` — clean dependency injection for passing app state into tools without globals

---

## liteagent vs Anthropic Claude Agent SDK

[Claude Agent SDK](https://platform.claude.com/docs/en/agent-sdk/overview) (~4.9k GitHub stars) is
Anthropic's agent library — essentially Claude Code exposed as a programmable Python/TypeScript package.

### Fundamental difference

Claude Agent SDK is a **black box wrapper around Claude Code**. You don't implement tools or the loop.
You call `query()`, Claude executes built-in tools (Read, Edit, Bash, etc.) internally, and you get
back an async iterator of messages. liteagent is the opposite — the loop is fully explicit and
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

| liteagent feature | Why it matters |
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

## liteagent vs Pydantic AI

[Pydantic AI](https://ai.pydantic.dev/) (~15k GitHub stars) is from the Pydantic team. Their pitch:
"Bring that FastAPI feeling to GenAI app development." Type-safe, model-agnostic, production-grade.

### Fundamental difference

Pydantic AI agents are **declarative and stateless** — you define an `Agent[DepsType, OutputType]`,
call `run()` with inputs, get a result back. liteagent agents are **stateful** — they hold message
history, streaming state, and queues. You interact via `prompt()`, `steer()`, `follow_up()`, `abort()`.
Pydantic AI has no concept of steering or follow-ups.

### What Pydantic AI has that we don't (and whether we care)

| Pydantic AI feature | Our stance |
|---|---|
| `@agent.tool` decorator (auto schema from type hints + docstrings) | Want this — Phase 4 TODO (same as OpenAI SDK's `@function_tool`) |
| Structured output (`output_type` → Pydantic model, with self-correcting validation) | Their best feature. Could add later. |
| Dependency injection (`RunContext[DepsType]`, type-safe) | Clean pattern — consider for Phase 4 |
| `FallbackModel` (auto model failover) | Nice. Consumer can implement externally for now. |
| History processors (pluggable message history modification) | We have `transform_context` — equivalent concept. |
| Output validators with `ModelRetry` (validation failures fed back to LLM) | Clever self-correction loop. Could add later. |
| Three output modes (tool-based, native, prompted) | Adapts to model capabilities. Sophisticated. |
| Logfire/OpenTelemetry observability | Don't need — consumer's problem |
| Graph-based workflows (typed state machines) | Don't need — sub-agents via tools works fine |
| Durable execution (Temporal, DBOS, Prefect) | Don't need — consumer's problem |
| MCP + A2A protocol support | Could add later |

### What we have that Pydantic AI doesn't

| liteagent feature | Why it matters |
|---|---|
| **Steering (mid-run interruption)** | User can redirect agent while it's executing tools. Pydantic AI has no equivalent — `run()` is fire-and-forget. |
| **Follow-up messages** | Queue messages that wait until agent finishes. Pydantic AI doesn't support this. |
| **Streaming tool output (`on_update`)** | Tools can push partial results (e.g., bash output line-by-line). Pydantic AI tools return all-or-nothing. |
| **Cancellation signal to tools** | `asyncio.Event` propagated through entire chain. Pydantic AI tools can't be cancelled mid-execution. |
| **Stateful agent** | Message history persists across calls. Pydantic AI requires passing `message_history=` manually each time. |
| **Dual while-loop architecture** | Inner loop (tools + steering), outer loop (follow-ups). Enables complex interaction patterns. |
| **Sequential tool execution** | Enables steering between tools. Pydantic AI tool execution model doesn't support interruption. |
| **Minimal core** | Small, hackable codebase (5 files). Pydantic AI is a large framework. |

### Different philosophies

- **Pydantic AI**: "FastAPI for AI" — type-safe, batteries-included, production-grade, backed by VC-funded company. Optimized for structured output and observability. Declarative agents, stateless runs.
- **liteagent**: "Here's the core loop, bring your own everything" — minimal, stateful, hackable. Optimized for real-time interaction (steering, follow-ups, streaming). You understand every line.

### Ideas to steal from Pydantic AI

- Self-correcting structured output — validation failures fed back to LLM with specific error message, creating a reflection loop. Smartest structured output approach of any framework.
- `@agent.tool` with docstring parsing (Google, NumPy, Sphinx formats) — richer schema generation than just type hints.
- `ModelRetry` exception — clean pattern for tools to request LLM retry with feedback.
- `FallbackModel` — automatic provider failover. Simple concept, high value for production.

---

## Philosophy Comparison (all four)

| | Claude Agent SDK | OpenAI Agents SDK | Pydantic AI | liteagent |
|---|---|---|---|---|
| **Approach** | Claude Code as a library | Multi-agent framework | "FastAPI for AI" | Core Python agent loop |
| **Agent loop** | Black box | Visible but managed | Internal (graph iteration available) | Fully explicit (ported from pi-mono) |
| **Tools** | Built-in | Decorators + hosted | Decorated functions, auto-schema | Explicit dataclass + execute |
| **Models** | Claude only | OpenAI-first | 20+ providers (own abstractions) | Any (litellm) |
| **Structured output** | JSON Schema | Pydantic model | First-class (3 modes + self-correction) | None (could add later) |
| **Steering** | None | None | None | First-class (ported from pi-mono) |
| **Streaming tools** | None | None | None | `on_update` callback |
| **Lock-in** | Claude | OpenAI | None (but large framework) | None |
| **Target user** | "Claude Code in my app" | "Multi-agent system fast" | "Type-safe AI apps in production" | "Understand the loop, build with it" |
| **Size** | Medium | High | Large | 5 core files (litellm does the rest) |
