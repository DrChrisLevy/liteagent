# Types Deep Dive

Understanding every type in py-pi-agent — what it is, why it exists, and where it's used.

## The Big Picture

The types form three layers, from bottom to top:

```
┌─────────────────────────────────────────────┐
│  Agent Layer (agent.py)                     │
│  AgentState — snapshot of the running agent │
├─────────────────────────────────────────────┤
│  Loop Layer (loop.py)                       │
│  AgentContext — state the loop operates on  │
│  AgentConfig — hooks the loop calls         │
│  Events — what the loop emits               │
├─────────────────────────────────────────────┤
│  Tool Layer                                 │
│  Tool — what the LLM can call               │
│  ToolResult — what comes back               │
├─────────────────────────────────────────────┤
│  Foundation                                 │
│  StopReason — why the LLM stopped           │
│  Messages — conversation history (dicts)    │
│  EventStream — already built (stream.py)    │
└─────────────────────────────────────────────┘
```

---

## 1. StopReason

**What:** Why did the LLM stop generating?

```python
StopReason = Literal["stop", "tool_calls", "length", "error", "aborted"]
```

| Value | Meaning | Who sets it | Loop behavior |
|---|---|---|---|
| `"stop"` | LLM finished naturally | litellm (mapped from provider) | Agent ends |
| `"tool_calls"` | LLM wants to call tools | litellm (mapped from provider) | Inner loop continues |
| `"length"` | Hit max token limit | litellm | Agent ends |
| `"error"` | LLM request failed | Our loop | Agent ends immediately |
| `"aborted"` | User cancelled | Our loop (checks `signal.is_set()`) | Agent ends immediately |

**Pi equivalent:** `StopReason` in `ai/src/types.ts` — same values except pi uses `"toolUse"` where we use `"tool_calls"` (litellm normalizes to OpenAI's naming).

**Where used:**
- On every assistant message (`msg["stop_reason"]`)
- Loop checks it to decide: continue (tool_calls) vs stop (everything else)
- Agent class checks it for error state

---

## 2. Tool

**What:** A tool the LLM can call. Combines the LLM-facing definition with the execution function.

```python
@dataclass
class Tool:
    name: str                    # LLM sees this ("read_file", "bash")
    description: str             # LLM sees this
    parameters: dict             # JSON Schema — sent to litellm as tool definition
    label: str = ""              # Human-readable for UI ("Read File")
    params_model: type = None    # Optional Pydantic BaseModel for validation
    execute: Callable = None     # async def(tool_call_id, params, signal, on_update) -> ToolResult
```

**Pi equivalent:** `AgentTool` in `agent/src/types.ts` — extends the base `Tool` (from `ai/src/types.ts`) with `label` and `execute`.

**Key insight:** Pi has TWO tool types:
- `Tool` (ai layer) = just the definition (name, description, parameters schema)
- `AgentTool` (agent layer) = definition + `execute` + `label`

We collapse these into one `Tool` because we don't have a separate "ai" package. The `parameters` dict serves double duty:
1. Sent to litellm as the function's JSON Schema (what the LLM sees)
2. Used alongside `params_model` for validation (what the loop uses)

### params_model — Pydantic validation + coercion

Optional Pydantic `BaseModel` class that validates and coerces tool arguments before `execute`
is called. Solves the problem of LLMs making type mistakes (sending `"42"` string when schema
says integer).

**Two things, two audiences:**
- `parameters` (JSON Schema dict) → tells the **LLM** what args to send
- `params_model` (Pydantic class) → tells the **loop** how to validate

The validation flow:
1. LLM returns args as JSON string → loop parses to dict
2. If `params_model` exists → `ParamsModel(**args_dict)` validates + coerces
3. Validated dict passed to `execute()`
4. If validation fails → error goes back to LLM as tool result with `is_error=True`

If `params_model` is `None`, validation is skipped and raw parsed args pass through.

**On validation errors:** There's no retry logic in the loop. The error text goes back to the
LLM as a tool result, and the LLM decides what to do — retry with fixed args, try a different
approach, or tell the user it failed. The LLM *is* the retry logic.

Phase 4 has a `@tool` decorator planned that will auto-generate both `parameters` and
`params_model` from Python type hints.

**The execute signature:**

```python
async def execute(
    tool_call_id: str,       # unique ID from LLM — links call to result
    params: dict,            # validated + coerced arguments
    signal: asyncio.Event,   # cancellation — check signal.is_set()
    on_update: Callable,     # streaming partial results to UI
) -> ToolResult:
```

- `tool_call_id` — the LLM generates this. We pass it through so events can be routed (UI knows which tool panel to update).
- `params` — already validated by Pydantic if `params_model` exists, otherwise raw parsed JSON.
- `signal` — `asyncio.Event` for cancellation. When user calls `agent.abort()`, it sets this event. Long-running tools should check `signal.is_set()` periodically and bail. Fast tools (like `echo`) never bother — they finish before anyone could abort.
- `on_update` — callback to push `tool_execution_update` events mid-execution. Without it, consumers only see `tool_execution_start` and `tool_execution_end` — nothing in between. Tools like `bash` use it to stream output line-by-line so the UI updates in real time.

Both `signal` and `on_update` default to `None`, so simple tools can ignore them entirely:
`async def echo(tool_call_id, params, **_)`.

**All tools are async.** Same as pi. Tools naturally do async things (subprocess, file I/O, network, sleep for cancellation checks). Even trivial tools are `async def` — the overhead is negligible and it keeps the interface uniform.

**Where used:**
- `AgentContext` holds the list of tools (snapshot for the current run)
- Loop builds `tool_schemas` from context.tools for litellm
- Loop calls `tool.execute()` when LLM makes a tool call
- Agent class exposes tools in `AgentState`

---

## 3. ToolResult

**What:** What a tool returns after execution.

```python
@dataclass
class ToolResult:
    content: list        # [{"type": "text", "text": "..."}, {"type": "image_url", ...}]
    details: dict = None # UI-only extras (not sent to LLM)
```

**Pi equivalent:** `AgentToolResult` in `agent/src/types.ts`.

**The content/details split:**

The LLM only understands text and images. But the UI might want richer data (interactive charts, syntax-highlighted code, etc.). So:

- `content` → goes to the LLM as the tool result message
- `details` → goes to the UI via events, never to the LLM

```python
# Example: a chart tool
ToolResult(
    content=[
        {"type": "text", "text": "Generated: Sales Chart"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}},
    ],
    details={"plotly_html": "<div>interactive version</div>"},
)
```

**Error handling note:** `ToolResult` does NOT have an `is_error` field. Tools signal errors by **raising exceptions**. The loop catches them, wraps the error message into a ToolResult, and sets `is_error=True` on the **tool result message** in the conversation. One path, one place where is_error lives.

**Where used — ToolResult flows to two places:**

1. **The conversation** (content only, for the LLM): Loop takes `ToolResult.content` and builds
   a tool result message dict (`{"role": "tool", "tool_call_id": ..., "content": ..., "is_error": ...}`).
   This gets appended to `context.messages` so the LLM sees what the tool returned.

2. **The event stream** (everything, for the UI): Loop emits a `tool_execution_end` event with
   the full `ToolResult` including `details`. The UI gets rich metadata for rendering; the LLM
   only gets text + images.

Also used by `on_update` for streaming partial results mid-execution — same type, just
intermediate snapshots that become `tool_execution_update` events.

---

## 4. AgentContext

**What:** The data snapshot the loop operates on. Created once at the start of each run.

```python
@dataclass
class AgentContext:
    system_prompt: str                  # loop reads, never mutates
    messages: list                      # loop APPENDS new messages here
    tools: list[Tool] | None = None    # loop reads, never mutates
```

**Pi equivalent:** `AgentContext` in `agent/src/types.ts` — identical purpose.

**Why separate from config?** Clean state vs behavior separation:
- **Context = "what the loop works with"** — the data it reads and mutates (messages) or just reads (tools, system_prompt). Maps naturally to what gets sent to the LLM.
- **Config = "how the loop behaves"** — hooks, model string, LLM parameters. These are wiring, not state.

**Mutability:** `system_prompt` and `tools` are read-only during the run. `messages` is the one field the loop mutates — it appends assistant messages and tool result messages as the run progresses.

**Snapshot:** The Agent class snapshots its current state into a context when `prompt()` is called. The loop works exclusively with this snapshot. Even if `agent.set_tools()` is called externally mid-run, the running loop won't see it (same as pi).

**Where used:**
- Created by Agent class at start of each `prompt()` / `continue_run()`
- Passed to `agent_loop()` → `run_loop()` → `stream_llm_response()` → `execute_tool_calls()`
- Loop appends new messages to `context.messages`
- Loop reads `context.tools` for tool schemas and execution
- Loop reads `context.system_prompt` for LLM calls

---

## 5. AgentConfig

**What:** Configuration and hooks for the loop. This is how consumers customize behavior.

```python
@dataclass
class AgentConfig:
    model: str                               # litellm model string
    convert_to_llm: Callable                 # REQUIRED: filter messages for LLM
    transform_context: Callable = None       # compaction, pruning
    get_steering_messages: Callable = None   # check for interruption
    get_follow_up_messages: Callable = None  # check for queued messages
    reasoning_effort: str = None             # "low"/"medium"/"high" or None
    max_tokens: int = None
    temperature: float = None
    max_retry_delay_ms: int = 60000
```

**Pi equivalent:** `AgentLoopConfig` in `agent/src/types.ts`.

**The hooks explained:**

### convert_to_llm (REQUIRED)
Transforms our internal message list into what litellm accepts. Standard messages (user, assistant, tool) pass through. Custom message types (like pi's `bashExecution` or summary messages) get converted to user messages or filtered out.

**Critical rule:** Must pass `thinking_blocks` and `reasoning_content` through on assistant messages. Anthropic requires thinking blocks from previous turns in subsequent requests.

**Real-world example — OpenAI multimodal workaround:** OpenAI tool messages only support string content. If a tool returns images (e.g. a chart), OpenAI silently drops them. Anthropic and Gemini handle `image_url` in tool results natively. A provider-aware `convert_to_llm` can strip images from tool results and re-inject them as a synthetic user message — the LLM still sees the images, just via a different message type. This is a semantic provider limitation that litellm doesn't normalize, so it's the hook's job.

For a simple agent with no custom types or provider workarounds, it can be an identity function: `convert_to_llm=lambda msgs: msgs`.

### transform_context (optional, async)
Called before every LLM call. Can modify the message list — e.g., compaction (summarizing old messages to fit context window), injecting dynamic system context, pruning images from old turns.

### get_steering_messages (optional, async)
Called after each tool execution. Returns messages the user wants to inject NOW, interrupting the current tool sequence. If messages returned → skip remaining tools, feed these to LLM instead.

### get_follow_up_messages (optional, async)
Called when the agent would otherwise stop (no more tool calls). Returns messages queued for "after you're done." If messages returned → outer loop continues.

### Sync vs async hooks

All four hooks can be sync or async (same as pi). The loop `await`s them uniformly —
if a sync function is returned, `await` resolves it immediately. This matches pi's
pattern: `convertToLlm: (messages) => Message[] | Promise<Message[]>`.

In practice: `convert_to_llm` is usually sync (just filtering a list), while
`transform_context` is usually async (might call an LLM for summarization). But the
loop doesn't care — it handles both.

**Where used:**
- Loop calls all four hooks at specific points in the dual loop
- Agent class creates an AgentConfig and injects its queue-backed implementations of get_steering_messages and get_follow_up_messages

---

## 6. Events

**What:** Plain dicts emitted by the loop via EventStream. The contract between the loop and any UI/consumer.

Events are NOT a class. They're dicts with a `"type"` key. This matches pi (plain objects) and makes JSON serialization trivial.

### Agent lifecycle
```python
{"type": "agent_start"}
{"type": "agent_end", "messages": [...]}  # only NEW messages from this run
```
Emitted once per `prompt()` / `continue_run()` call. `agent_end.messages` contains only the messages added during this run, not the full history.

### Turn lifecycle
```python
{"type": "turn_start"}
{"type": "turn_end", "message": ..., "tool_results": [...]}
```
One turn = one LLM call + its tool executions. Multiple turns happen when tools are called (inner loop iterates).

### Message lifecycle
```python
{"type": "message_start", "message": ...}
{"type": "message_update", "message": ..., "delta": ..., "delta_type": ...}
{"type": "message_end", "message": ...}
```
- `message_start` / `message_end` — emitted for ALL message types (user, assistant, tool result, steering)
- `message_update` — assistant only, during streaming. Contains the raw litellm delta.

### delta_type (convenience field)
```
"text_delta"      — delta.content was present (LLM generating text)
"thinking_delta"  — delta.reasoning_content was present (reasoning tokens)
"tool_call_delta" — delta.tool_calls was present (building tool call)
```

This saves consumers from inspecting the delta object to figure out what changed.

### Tool execution lifecycle
```python
{"type": "tool_execution_start", "tool_call_id": ..., "tool_name": ..., "args": ...}
{"type": "tool_execution_update", "tool_call_id": ..., "tool_name": ..., "partial": ...}
{"type": "tool_execution_end", "tool_call_id": ..., "tool_name": ..., "result": ..., "is_error": bool}
```
Separate from message events. These track the actual execution of tool code, not the LLM's tool call request.

**Pi equivalent:** `AgentEvent` union type in `agent/src/types.ts` — same event types, camelCase naming.

**Where used:**
- Pushed to EventStream by the loop
- Consumed by `async for event in stream` in any framework
- Agent class subscribes internally to update `AgentState`

---

## 7. AgentState

**What:** Read-only snapshot of the agent's current state.

```python
@dataclass
class AgentState:
    system_prompt: str
    model: str
    thinking_level: str              # "off", "low", "medium", "high"
    tools: list[Tool]
    messages: list                   # full message history
    is_streaming: bool               # True while loop is running
    stream_message: dict | None      # partial message being streamed
    pending_tool_calls: set[str]     # tool call IDs currently executing
    error: str | None                # last error message
```

**Pi equivalent:** `AgentState` in `agent/src/types.ts` — nearly identical.

**thinking_level vs reasoning_effort:** `thinking_level` is the user-facing setting on AgentState. It maps to `reasoning_effort` on AgentConfig:
- `"off"` → `reasoning_effort=None` (don't send to litellm)
- `"none"`, `"minimal"`, `"low"`, `"medium"`, `"high"`, `"xhigh"` → passed to litellm's `reasoning_effort` parameter

litellm supports all 6 values and maps them to provider-specific budgets internally:
- Anthropic: `reasoning_effort="low"` → `thinking={"type": "enabled", "budget_tokens": 1024}`
- Gemini: maps to `thinkingBudget` or `thinkingLevel` depending on model version
- OpenAI: `"xhigh"` supported for gpt-5.2+
- DeepSeek: only supports enabled/disabled, ignores budget

Pi does explicit budget math (1024-16384 tokens per level). We don't need to — litellm
handles it. No `ThinkingBudgets` type needed.

**Where used:**
- Agent class maintains it internally
- Exposed via `agent.state` property (read-only) — e.g. UI can show which tools are available
- Updated by subscribing to the EventStream (agent_start → is_streaming=True, tool_execution_start → add to pending_tool_calls, etc.)

### set_tools and dynamic capabilities

The Agent class has `set_tools()` (and `set_model()`, `set_system_prompt()`, etc.) to
reconfigure between runs. Can't be called mid-run (blocked by `is_streaming` guard).

Pi's coding agent uses this extensively — it has a **tool registry** (all discovered tools)
and an **active subset**. Extensions dynamically toggle tools on/off:
- Plan mode: start read-only, enable write tools after planning
- Presets: save/load combos of model + thinking level + tool set
- Permission escalation: add dangerous tools only after user confirms

The primitive is simple (just a setter + snapshot pattern), but it enables a full dynamic
capability system. We don't need the extension framework for our core library, but the
primitive makes it possible for consumers to build one.

---

## 8. Messages (plain dicts)

Messages are NOT a type/class — they're plain dicts following litellm's OpenAI-compatible format. This is deliberate: litellm expects dicts, and wrapping them would create a parallel type system.

### User message
```python
{"role": "user", "content": "Fix the bug", "timestamp": 1708531200000}
```

### Assistant message (our canonical format)
```python
{
    "role": "assistant",
    "content": "Here's the fix..." | None,
    "tool_calls": [...] | None,
    "thinking_blocks": [...] | None,       # Anthropic only
    "reasoning_content": "..." | None,     # all providers
    "usage": {"prompt_tokens": ..., "completion_tokens": ..., ...},
    "stop_reason": "stop" | "tool_calls" | "length" | "error" | "aborted",
    "timestamp": 1708531200000,
}
```

### Tool result message
```python
{
    "role": "tool",
    "tool_call_id": "call_123",
    "content": [{"type": "text", "text": "file contents"}],
    "is_error": False,
    "timestamp": 1708531200000,
}
```

**Pi equivalent:** `Message` union type in `ai/src/types.ts` — `UserMessage | AssistantMessage | ToolResultMessage`. Pi uses classes/interfaces; we use dicts for litellm compat.

---

## How They All Connect

```
Consumer calls agent.prompt("Fix the bug")
    │
    ├─ Agent snapshots AgentContext (system_prompt, messages, tools)
    ├─ Agent creates AgentConfig (with hooks wired to its queues)
    ├─ Agent creates asyncio.Event (signal)
    ├─ Agent calls agent_loop(messages, context, config, signal)
    │       │
    │       ├─ Returns EventStream immediately
    │       └─ Spawns async task running run_loop()
    │               │
    │               ├─ Calls config.get_steering_messages()  ← AgentConfig hook
    │               ├─ Calls config.transform_context()      ← AgentConfig hook
    │               ├─ Calls config.convert_to_llm()         ← AgentConfig hook
    │               ├─ Builds tool_schemas from context.tools ← AgentContext data
    │               ├─ Calls litellm.acompletion()
    │               │       └─ Streams chunks → message_update events with delta_type
    │               ├─ Builds assistant message (dict) with StopReason, usage
    │               ├─ Appends assistant message to context.messages
    │               ├─ If stop_reason == "tool_calls":
    │               │       ├─ For each tool_call:
    │               │       │   ├─ Find Tool by name in context.tools
    │               │       │   ├─ Validate params (Pydantic if params_model)
    │               │       │   ├─ Call tool.execute() → ToolResult
    │               │       │   ├─ Build tool result message (dict)
    │               │       │   └─ Check steering after each tool
    │               │       └─ Inner loop continues
    │               └─ If stop_reason != "tool_calls":
    │                       ├─ Check follow-ups → outer loop
    │                       └─ Emit agent_end
    │
    └─ Consumer iterates: async for event in stream
            └─ Events are plain dicts with "type" key
```

---

## Pi Types We DON'T Need

| Pi type | Why we skip it |
|---|---|
| `Api` / `KnownApi` / `Provider` / `KnownProvider` | litellm handles provider routing |
| `StreamFn` / `StreamFunction` / `StreamOptions` / `SimpleStreamOptions` | litellm replaces the entire streaming layer |
| `ThinkingBudgets` | litellm maps reasoning_effort → provider budgets internally (Anthropic: budget_tokens, Gemini: thinkingBudget). Confirmed in litellm source. |
| `getApiKey` hook | Dynamic API key resolution per LLM call (e.g., expiring OAuth tokens). litellm has its own key management. Revisit if OAuth rotation needed. |
| `TextContent` / `ThinkingContent` / `ImageContent` / `ToolCall` | We use litellm's OpenAI-format dicts |
| `AssistantMessageEvent` (text_start/delta/end, etc.) | Our event system is simpler — just `message_update` with `delta_type` |
| `Context` (ai layer) | Replaced by our `AgentContext` — same role, Python idioms |
| `CustomAgentMessages` | TypeScript declaration merging pattern — Python doesn't need this |
| `OpenAICompletionsCompat` / `OpenRouterRouting` / etc. | Provider compatibility — litellm's problem |
| `Model` | Model metadata/catalog — litellm's problem |

Pi built ~6,800 lines of provider-specific code. litellm replaces ALL of it, which is why our types layer is so much smaller.

---

## Summary: What We'll Build in types.py

```python
# types.py — the whole file

StopReason = Literal["stop", "tool_calls", "length", "error", "aborted"]

@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    label: str = ""
    params_model: type = None
    execute: Callable = None

@dataclass
class ToolResult:
    content: list
    details: dict = None

@dataclass
class AgentContext:
    system_prompt: str
    messages: list
    tools: list[Tool] | None = None

@dataclass
class AgentConfig:
    model: str
    convert_to_llm: Callable
    transform_context: Callable = None
    get_steering_messages: Callable = None
    get_follow_up_messages: Callable = None
    reasoning_effort: str = None
    max_tokens: int = None
    temperature: float = None
    max_retry_delay_ms: int = 60000

@dataclass
class AgentState:
    system_prompt: str
    model: str
    thinking_level: str       # "off", "none", "minimal", "low", "medium", "high", "xhigh"
    tools: list[Tool]
    messages: list
    is_streaming: bool
    stream_message: dict | None
    pending_tool_calls: set[str]
    error: str | None
```

That's it. Five dataclasses, one Literal type, and events are plain dicts. Small surface area, clear contracts.
