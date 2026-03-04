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
| **Minimal core** | ~1,230 lines total (~335 lines of loop logic + ~265 lines of streaming). OpenAI SDK is a much larger surface area. |

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

## py-pi-agent vs Pydantic AI

[Pydantic AI](https://ai.pydantic.dev/) (~15k GitHub stars) is from the Pydantic team. Their pitch:
"Bring that FastAPI feeling to GenAI app development." Type-safe, model-agnostic, production-grade.

### Fundamental difference

Pydantic AI agents are **declarative and stateless** — you define an `Agent[DepsType, OutputType]`,
call `run()` with inputs, get a result back. py-pi-agent agents are **stateful** — they hold message
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

| py-pi-agent feature | Why it matters |
|---|---|
| **Steering (mid-run interruption)** | User can redirect agent while it's executing tools. Pydantic AI has no equivalent — `run()` is fire-and-forget. |
| **Follow-up messages** | Queue messages that wait until agent finishes. Pydantic AI doesn't support this. |
| **Streaming tool output (`on_update`)** | Tools can push partial results (e.g., bash output line-by-line). Pydantic AI tools return all-or-nothing. |
| **Cancellation signal to tools** | `asyncio.Event` propagated through entire chain. Pydantic AI tools can't be cancelled mid-execution. |
| **Stateful agent** | Message history persists across calls. Pydantic AI requires passing `message_history=` manually each time. |
| **Dual while-loop architecture** | Inner loop (tools + steering), outer loop (follow-ups). Enables complex interaction patterns. |
| **Sequential tool execution** | Enables steering between tools. Pydantic AI tool execution model doesn't support interruption. |
| **Minimal core** | ~1,230 lines total. Pydantic AI is a large framework (thousands of lines). |

### Different philosophies

- **Pydantic AI**: "FastAPI for AI" — type-safe, batteries-included, production-grade, backed by VC-funded company. Optimized for structured output and observability. Declarative agents, stateless runs.
- **py-pi-agent**: "Here's the core loop, bring your own everything" — minimal, stateful, hackable. Optimized for real-time interaction (steering, follow-ups, streaming). You understand every line.

### Ideas to steal from Pydantic AI

- Self-correcting structured output — validation failures fed back to LLM with specific error message, creating a reflection loop. Smartest structured output approach of any framework.
- `@agent.tool` with docstring parsing (Google, NumPy, Sphinx formats) — richer schema generation than just type hints.
- `ModelRetry` exception — clean pattern for tools to request LLM retry with feedback.
- `FallbackModel` — automatic provider failover. Simple concept, high value for production.

---

## py-pi-agent vs pi-mono (our origin)

[pi-mono](https://github.com/badlogic/pi-mono) is the TypeScript agent framework py-pi-agent is ported
from. Same author (pi/badlogic), battle-tested in production. This isn't a comparison of competitors —
it's a comparison of the original and its Python translation.

### What we're faithfully porting

| pi-mono pattern | py-pi-agent equivalent |
|---|---|
| Dual while-loop (`runLoop()` in `agent-loop.ts`) | Same architecture in `loop.py` |
| `EventStream<T, R>` async queue | `EventStream` in `stream.py` |
| `AgentTool` with `execute(id, params, signal, onUpdate)` | `Tool` with `execute(id, params, signal, on_update)` |
| Steering queue (checked after each tool) | Same — `steer()` with dequeue after each tool |
| Follow-up queue (checked when agent would stop) | Same — `follow_up()` checked at outer loop boundary |
| `transformContext` hook (AgentMessage[] → AgentMessage[]) | `transform_context` hook |
| `convertToLlm` hook (AgentMessage[] → Message[]) | `convert_to_llm` hook |
| AbortSignal propagated to tools | `asyncio.Event` signal propagated to tools |
| Sequential tool execution (enables steering) | Same design decision |
| Stateful `Agent` class wrapping stateless loop functions | Same two-layer design |
| Event types: message_start/update/end, tool_execution_start/update/end, turn_start/end, agent_start/end | Same event taxonomy |

### What we're doing differently

| Difference | pi-mono (TypeScript) | py-pi-agent (Python) |
|---|---|---|
| **LLM interface** | Hand-rolled providers (~6,800 lines across OpenAI, Anthropic, Google, etc.) | litellm — one `acompletion()` call handles all providers |
| **Validation** | AJV (JSON Schema validation) | Pydantic — validates AND coerces types (e.g., `"42"` → `42`) |
| **Proxy streaming** | Built-in proxy mode with bandwidth-optimized SSE reconstruction | Not needed — consumer handles transport |
| **Session caching** | `sessionId` for provider-specific prompt cache hints | Not implementing — litellm handles this if needed |
| **Transport abstraction** | SSE / WebSocket / auto toggle | Not needed — consumer picks transport |
| **Thinking budgets** | Token-based thinking level configuration (minimal/low/medium/high) | litellm passes thinking config through natively |
| **Dynamic API keys** | `getApiKey(provider)` callback before each LLM call (for OAuth) | litellm handles auth; consumer can override via config |
| **Concurrency** | AbortController / AbortSignal (Web API) | asyncio.Event + asyncio cancellation (Python native) |

### Why litellm changes the game

pi-mono's `packages/ai/` directory contains ~6,800 lines of provider-specific streaming code:
per-provider message conversion, chunk parsing, error handling, retry logic. All of that is replaced
by a single litellm dependency in py-pi-agent. This is the biggest architectural simplification —
it means our core loop logic is ~335 lines (plus ~265 lines of streaming chunk handling in `stream_llm_response`) instead of needing thousands of lines of provider glue.

The tradeoff: we depend on litellm's correctness and maintenance. But litellm is actively maintained,
widely used, and covers 100+ providers. The bet is worth it.

### What we're NOT porting (and why)

| pi-mono feature | Why we skip it |
|---|---|
| Proxy streaming mode | Solves browser→server routing. Python consumers don't need this — they run server-side. |
| Transport abstraction (SSE/WebSocket) | Consumer's problem. We emit events, they choose transport. |
| `sessionId` provider hint | litellm handles provider-specific features. |
| `getApiKey` callback | litellm handles auth. Consumer can swap keys via litellm's API key config. |
| Bandwidth-optimized delta reconstruction | Specific to browser streaming. Not relevant for Python library. |

### Fidelity scorecard (March 2026)

How close is our implementation to pi-mono's, component by component:

| Component | Fidelity | Notes |
|-----------|----------|-------|
| **Dual while loop** | 99% | Structurally identical. Same control flow, same edge cases, same event ordering. |
| **Tool execution** | 98% | Same sequential model, steering interrupt, skip semantics, argument validation. |
| **Agent class** | 95% | All core methods. Missing pi-specific options (sessionId, streamFn, transport — see litellm section below). |
| **Event system** | 95% | Same 10 event types, same ordering. Dict-based vs pi's typed unions. |
| **Streaming** | 85% | Works, but we handle raw litellm chunks ourselves (~265 lines) where pi delegates to pi-ai's parsed events. No injectable streamFn. |
| **Type system** | 70% | Intentionally different — plain dicts vs pi's typed message classes. No `CustomAgentMessages` declaration merging. |
| **EventStream** | 90% | Single-consumer (vs pi's multi-consumer). Fine — Agent class is the sole consumer; externals use subscribe(). |

### What litellm already handles (pi features that aren't gaps)

Pi builds several features into its streaming layer that litellm provides out of the box.
These look like "missing features" when comparing code, but they're handled by our dependency:

| Pi feature | litellm equivalent | Gap? |
|---|---|---|
| `getApiKey(provider)` — dynamic per-call API key for expiring OAuth tokens | `acompletion(api_key=...)` per call; also supports `api_key="os.environ/VAR"` for dynamic resolution | **No** — ~3 lines to add if needed |
| `thinkingBudgets` — custom token budgets per thinking level | `reasoning_effort` param mapped to provider-specific budgets automatically (Gemini → `thinkingBudget`, Anthropic → pass-through, OpenAI → pass-through) | **No** — already handled |
| `maxRetryDelayMs` — cap on server-requested retry waits | Exponential backoff via tenacity with hardcoded 10s max. `num_retries` passed through. | **No** — already handled |
| `transport` — SSE vs WebSocket selection | Not applicable — litellm uses httpx for all LLM calls. Transport is a pi/browser concept. | **N/A** |
| Cost tracking | `litellm.include_cost_in_streaming_usage = True` + `completion_cost()`. Available, just not wired. | **Easy add** (~5 lines) |
| `sessionId` — provider-specific session caching (OpenAI Codex) | litellm doesn't pass session IDs to providers | **Minor gap** — only one provider uses it |
| `streamFn` — injectable stream function for proxy backends | litellm already supports 100+ providers, custom base URLs, proxies | **Low priority** — reduced need |

### Design decisions vs pi

| Decision | Pi's approach | Our approach | Why |
|---|---|---|---|
| **Messages** | Typed union: `UserMessage \| AssistantMessage \| ToolResultMessage` + `CustomAgentMessages` via declaration merging | Plain dicts with `role` field | litellm speaks Chat Completions dicts. Typed classes would mean constant dict↔class conversion at the litellm boundary. Dicts are idiomatic for this layer. |
| **EventStream consumers** | Multi-consumer via waiters array | Single-consumer `asyncio.Queue` | Agent class is the sole stream reader (same as pi in practice). External consumers use `subscribe()`. No need for multi-consumer complexity. |
| **Streaming events** | Granular: `text_start`, `text_delta`, `text_end`, `thinking_start`/`delta`/`end`, `toolcall_start`/`delta`/`end` | Unified: `message_update` with `delta_type` field (`text_delta`, `thinking_delta`, `tool_call_delta`) | Same information, different shape. Pi's granularity is useful for its UI framework. Our consumers check `delta_type` instead — simpler, same expressiveness. |
| **Usage shape** | `{input, output, cacheRead, cacheWrite, totalTokens, cost: {...}}` | `{prompt_tokens, completion_tokens, total_tokens, cache_read_tokens, cache_creation_tokens}` (litellm naming, no cost yet) | We use litellm's field names directly to avoid mapping. Cost can be added via `litellm.include_cost_in_streaming_usage`. |
| **Model representation** | `Model<any>` object with id, name, api, provider, costs, context window, max tokens | Plain string (`"anthropic/claude-sonnet-4-6"`) | litellm resolves everything from the model string. Pi needs the object because it manages providers itself. |
| **Validation** | AJV + TypeBox (JSON Schema compiler, strict mode off, type coercion) | Pydantic BaseModel (optional per tool) | Pydantic validates AND coerces natively. Same role as AJV — Python's equivalent. |

---

## Philosophy Comparison (all five)

| | Claude Agent SDK | OpenAI Agents SDK | Pydantic AI | pi-mono | py-pi-agent |
|---|---|---|---|---|---|
| **Approach** | Claude Code as a library | Multi-agent framework | "FastAPI for AI" | Production TS agent loop | Core Python agent loop |
| **Agent loop** | Black box | Visible but managed | Internal (graph iteration available) | Fully explicit | Fully explicit (ported from pi-mono) |
| **Tools** | Built-in | Decorators + hosted | Decorated functions, auto-schema | Explicit interface + execute | Explicit dataclass + execute |
| **Models** | Claude only | OpenAI-first | 20+ providers (own abstractions) | Hand-rolled providers | Any (litellm) |
| **Structured output** | JSON Schema | Pydantic model | First-class (3 modes + self-correction) | None | None (could add later) |
| **Steering** | None | None | None | First-class | First-class (ported) |
| **Streaming tools** | None | None | None | `onUpdate` callback | `on_update` callback (ported) |
| **Lock-in** | Claude | OpenAI | None (but large framework) | None | None |
| **Target user** | "Claude Code in my app" | "Multi-agent system fast" | "Type-safe AI apps in production" | "Full control, production TS" | "Understand the loop, learn Python" |
| **Size** | Medium | High | Large | ~3,300 lines (5 core files) | ~1,230 lines (4 core files, litellm does the rest) |
