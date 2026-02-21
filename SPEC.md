# py-pi-agent — WIP Spec

A Python core agent loop library inspired by [pi-mono](https://github.com/badlogic/pi-mono)'s `packages/agent`. Framework-agnostic, plug-and-play, usable from FastHTML, FastAPI, Slack, CLI, or anything else.

## Dependencies

- `asyncio` — async runtime (stdlib)
- `litellm` — unified LLM interface (Chat Completions format for all providers)
- `pydantic` — tool argument validation + type coercion

## Project Structure

```
py_pi_agent/
    __init__.py
    stream.py       # EventStream (async producer-consumer queue)
    types.py        # Event types, config, tool protocol, stop reasons
    loop.py         # Dual while loop with streaming + steering
    agent.py        # Agent class — stateful wrapper around the loop
```

---

## Design Principles

- **Framework-agnostic**: The loop emits events. What consumes them (SSE, WebSocket, Slack, CLI) is the consumer's problem.
- **Minimal**: No tools, no system prompt, no UI baked in. All passed in by the consumer.
- **Async-native**: Built on asyncio. Tools are `async def`.
- **LLM-agnostic**: Works with any model litellm supports (Claude, GPT, Gemini, DeepSeek, etc.)
- **Inspired by pi**: Sequential tool execution, steering, follow-ups, event streaming — same architecture, Python idioms.

---

## EventStream

Async producer-consumer queue. The loop pushes events, consumers iterate.

```python
class EventStream:
    """Async event stream with producer-consumer pattern."""

    def push(self, event: dict):
        """Producer pushes events (non-blocking)."""

    def end(self):
        """Signal stream completion."""

    async def result(self) -> list:
        """Await final result (all messages)."""

    async def __aiter__(self):
        """Consumer iterates: async for event in stream: ..."""
```

Built on `asyncio.Queue`. Backpressure built in (if consumer is slow, events queue up).

### Usage from any framework

```python
from py_pi_agent import Agent

agent = Agent(model="claude-opus-4-6-20250219", tools=my_tools)
stream = agent.prompt("Fix the bug in main.py")

# FastHTML SSE endpoint
async for event in stream:
    yield sse(render_html(event))

# FastAPI WebSocket
async for event in stream:
    await ws.send_json(event)

# Slack bot
async for event in stream:
    if event["type"] == "message_end" and event["message"]["role"] == "assistant":
        slack.post_message(event["message"]["content"])

# CLI (rich/textual)
async for event in stream:
    if event["type"] == "message_update":
        delta = event["delta"]
        if delta["type"] == "text_delta":
            console.print(delta["text"], end="")
```

---

## Agent Class (stateful wrapper)

Pi has two layers: the raw loop functions and a stateful `Agent` class on top.
We do the same.

```python
class Agent:
    """Stateful agent that wraps the core loop."""

    def __init__(self, model, tools, system_prompt="", config=None):
        ...

    # --- Primary actions ---

    def prompt(self, message) -> EventStream:
        """Send a message and start the agent loop. Returns event stream.
        Throws if already streaming (use steer/follow_up instead)."""

    def continue_run(self) -> EventStream:
        """Resume from current context (e.g., after error recovery).
        Throws if already streaming or no messages to continue from.

        Special case when last message is assistant (pi edge case):
        1. Check steering queue first — if messages, run with those
        2. Else check follow-up queue — if messages, run with those
        3. Else throw (can't continue from assistant without new input)
        """

    # --- Mid-run message injection ---

    def steer(self, message):
        """Interrupt agent mid-run. Injected after current tool finishes,
        skips remaining tools. Immediate."""

    def follow_up(self, message):
        """Queue message for after agent finishes all tools.
        Deferred — only delivered when agent would otherwise stop."""

    # --- Control ---

    def abort(self):
        """Cancel current run. Sets stop_reason to 'aborted'."""

    async def wait_for_idle(self):
        """Await until agent finishes current run."""

    def reset(self):
        """Clear all messages, queues, and error state.
        Does NOT reset model, tools, or system prompt."""

    # --- Queue management ---

    def clear_steering_queue(self): ...
    def clear_follow_up_queue(self): ...
    def clear_all_queues(self): ...
    def has_queued_messages(self) -> bool: ...

    # --- Configuration ---

    def set_model(self, model): ...
    def set_system_prompt(self, prompt): ...
    def set_tools(self, tools): ...
    def set_thinking_level(self, level): ...

    # --- Event subscription ---

    def subscribe(self, callback) -> Callable:
        """Subscribe to agent events. Returns unsubscribe function."""

    # --- State access ---

    @property
    def state(self) -> AgentState:
        """Current agent state (read-only)."""

    @property
    def messages(self) -> list:
        """Current message history."""
```

### Agent State

```python
@dataclass
class AgentState:
    system_prompt: str
    model: str
    thinking_level: str             # "off", "minimal", "low", "medium", "high", "xhigh"
    tools: list[Tool]
    messages: list                  # full message history
    is_streaming: bool              # True while loop is running
    stream_message: dict | None     # current partial message being streamed
    pending_tool_calls: set[str]    # tool call IDs currently executing
    error: str | None               # last error message
```

### Concurrency Guard

```python
def prompt(self, message) -> EventStream:
    if self.state.is_streaming:
        raise RuntimeError(
            "Agent is already processing. Use steer() or follow_up() "
            "to queue messages, or await wait_for_idle()."
        )
    ...
```

### Steering & Follow-up Modes

```python
# "one-at-a-time" (default): dequeue one message per poll
# "all": dequeue all queued messages at once
agent.steering_mode = "one-at-a-time"
agent.follow_up_mode = "one-at-a-time"
```

---

## Event Types

> **Naming convention**: Pi uses camelCase (`toolUse`, `toolcall_delta`). We use snake_case
> (`tool_use`, `tool_call_delta`) per Python convention. Same semantics, different casing.

```python
# Agent lifecycle
{"type": "agent_start"}
{"type": "agent_end", "messages": [...]}

# Turn lifecycle (one LLM call + tool executions)
{"type": "turn_start"}
{"type": "turn_end", "message": ..., "tool_results": [...]}

# Message lifecycle
{"type": "message_start", "message": ...}
{"type": "message_update", "message": ..., "delta": ...}   # streaming tokens
{"type": "message_end", "message": ...}

# Streaming deltas (within message_update)
# delta.type can be:
#   "text_delta"      — text token from LLM
#   "thinking_delta"  — reasoning/thinking token
#   "tool_call_delta" — tool call arguments streaming
#
# Note: deltas are for UI display only. Tool execution uses the finalized
# assistant message from litellm, not reassembled deltas.

# Tool execution lifecycle
{"type": "tool_execution_start", "tool_call_id": ..., "tool_name": ..., "args": ...}
{"type": "tool_execution_update", "tool_call_id": ..., "tool_name": ..., "partial": ...}
{"type": "tool_execution_end", "tool_call_id": ..., "tool_name": ..., "result": ..., "is_error": bool}
```

---

## Tool Protocol

### Tool Definition (what the LLM sees + what the loop calls)

```python
@dataclass
class Tool:
    name: str               # LLM-facing name ("read_file", "bash")
    description: str        # LLM-facing description
    parameters: dict        # JSON Schema for arguments (sent to LLM)
    label: str = ""         # Human-readable UI label ("Read File")
    params_model: type = None  # Optional Pydantic BaseModel for validation + coercion
    execute: Callable = None   # async def(tool_call_id, params, signal, on_update) -> ToolResult
```

### Tool Result (what comes back)

```python
@dataclass
class ToolResult:
    content: list           # [TextBlock, ImageBlock, ...] — sent to LLM
    details: dict = None    # UI-only extras (not sent to LLM)
    # Note: is_error lives on the tool result MESSAGE (set by loop), not here.
    # Tools signal errors by throwing exceptions. Loop catches and marks is_error=True.
```

`content` vs `details` split: LLM sees text + images. UI gets extra metadata for rich rendering (e.g., plotly charts, syntax-highlighted code).

### Multimodal Tool Results

Tools can return text, images, or both. Content blocks use litellm/OpenAI format:

```python
# Text only
content = [{"type": "text", "text": "file contents here..."}]

# Image only (base64)
content = [{"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBOR..."}}]

# Mixed — text + multiple images (e.g., matplotlib plots, screenshots)
content = [
    {"type": "text", "text": "stdout:\nAnalysis complete. 3 charts generated."},
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBOR..."}},
    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,/9j/..."}},
]

# UI-only content goes in details (not sent to LLM)
details = {
    "plotly_html": ["<div>interactive chart</div>"],
    "syntax_highlighted": "<pre>...</pre>",
}
```

This follows the pattern from the agents/ project where `run_code` returns text + auto-captured
matplotlib/PIL images as content blocks, with plotly interactive charts in a separate UI-only channel.

The loop's `convert_to_llm` hook filters content to only LLM-compatible types (`text`, `image_url`).
Any custom types (plotly_html, etc.) should go in `details` so they're available to the UI but never
sent to the LLM.

### Tool Execute Signature

```python
async def my_tool(
    tool_call_id: str,               # unique ID from the LLM (for event routing + tracing)
    params: dict,                     # validated + coerced arguments
    signal: asyncio.Event = None,     # cancellation
    on_update: Callable = None,       # streaming partial results
) -> ToolResult:
    ...
```

The loop calls `execute(tool_call_id, validated_params, signal, on_update)` — same signature as pi.
`tool_call_id` is needed for:
- Routing `tool_execution_update` events to the right tool in the UI
- Linking tool results back to the assistant's tool call
- Tracing and auditing

### Tool Argument Validation

- Pydantic validation before calling execute
- Type coercion built-in (LLM sends "42" string → coerced to 42 int)
- Clear error messages back to LLM on validation failure
- Pydantic is Python's equivalent of AJV + TypeBox (what pi uses in TypeScript)

**Normalization path:**
1. Tool defines `parameters` as JSON Schema dict (for sending to LLM as tool definition)
2. Tool defines a Pydantic BaseModel for its params type
3. LLM returns args as JSON string → loop parses to dict → Pydantic validates + coerces
4. Validated params passed to `execute()` as a dict (not the Pydantic model instance)

If a tool only provides JSON Schema (no Pydantic model), validation is skipped and
raw parsed args are passed through — same as pi's browser fallback behavior.

### Tool Execution

- **Sequential** (same as pi) — one tool at a time
- After each tool: check steering queue for user interruption
- If steering: skip remaining tools, mark skipped as error with "Skipped due to queued user message"
- Tools receive `signal` for cancellation and `on_update` callback for streaming

---

## Dual Loop Architecture

```python
async def run_loop(context, config, stream, signal):
    pending_messages = await config.get_steering_messages()

    # OUTER LOOP: handles follow-up messages after agent settles
    while True:
        has_tool_calls = True

        # INNER LOOP: LLM calls + tool execution + steering
        while has_tool_calls or pending_messages:
            # 1. Inject steering/follow-up messages into context
            # 2. Stream LLM response (tokens flow to EventStream)
            # 3. Execute tools sequentially (check steering after each)
            # 4. Loop if more tool calls or steering

        # Agent would stop — check for queued follow-ups
        follow_ups = await config.get_follow_up_messages()
        if follow_ups:
            pending_messages = follow_ups
            continue

        break

    stream.push({"type": "agent_end"})
    stream.end()
```

### Why two loops?

- **Inner loop**: handles the LLM-tool cycle + immediate steering (interruption)
- **Outer loop**: handles deferred follow-ups (messages queued while agent was busy)
- Steering = "stop what you're doing, do this instead" (immediate)
- Follow-up = "when you're done, also do this" (deferred)

---

## Streaming LLM Response

```python
async def stream_llm_response(context, config, stream, signal):
    # Hook: transform context before sending (compaction, pruning)
    messages = await config.transform_context(context.messages)

    # Hook: convert to LLM format (filter out UI-only message types)
    llm_messages = config.convert_to_llm(messages)

    response = await litellm.acompletion(
        model=config.model,
        messages=llm_messages,
        tools=config.tool_schemas,
        stream=True,
        reasoning_effort=config.reasoning_effort,
    )

    # Stream 3 types of chunks
    async for chunk in response:
        delta = chunk.choices[0].delta

        if delta.reasoning_content:     # thinking tokens
            stream.push(thinking_delta_event)

        if delta.content:               # text tokens
            stream.push(text_delta_event)

        if delta.tool_calls:            # tool call fragments
            stream.push(tool_call_delta_event)

    # Build finalized assistant message from litellm response
    # This is the canonical internal format — stored in context, emitted in events
    assistant_msg = build_assistant_message(response, full_content, tool_calls, reasoning)
    context.messages.append(assistant_msg)
```

### Canonical Assistant Message Shape

Built from the finalized litellm response (not reassembled deltas):

```python
{
    "role": "assistant",
    "content": "response text" | None,              # None when only tool calls
    "tool_calls": [                                  # from litellm response
        {"id": "call_123", "type": "function",
         "function": {"name": "bash", "arguments": '{"command": "ls"}'}},
    ] | None,
    "thinking_blocks": [...] | None,                 # Anthropic only, includes signatures
    "reasoning_content": "thinking text" | None,     # universal, all providers
    "usage": {                                       # from litellm response
        "input": 1234, "output": 567,
        "cache_read": 890, "cache_write": 0,
        "total_tokens": 2691,
    },
    "stop_reason": "stop" | "tool_use" | "length" | "error" | "aborted",
}
```

This mirrors litellm's response format. The loop reads `tool_calls` to decide whether to
continue. Usage and stop_reason come directly from litellm. Thinking fields are preserved
for future messages (Anthropic requires them back).

---

## Config (Hook Points)

```python
@dataclass
class AgentConfig:
    model: str                              # litellm model string

    # REQUIRED: filter messages for LLM (remove UI-only types)
    convert_to_llm: Callable

    # OPTIONAL hooks
    transform_context: Callable = None      # compaction, pruning, injection
    get_steering_messages: Callable = None  # check for user interruption
    get_follow_up_messages: Callable = None # check for queued messages

    # LLM parameters
    reasoning_effort: str = None            # "off", "minimal", "low", "medium", "high", "xhigh"
    max_tokens: int = None
    temperature: float = None

    # Retry behavior
    max_retry_delay_ms: int = 60000         # cap server-requested retry delays
```

---

## Error Handling

### Tool Errors vs LLM Errors — different behavior

**Tool error** → loop CONTINUES:

Tools signal failure by **raising exceptions**. The loop catches and wraps:

```python
try:
    result = await tool.execute(tool_call_id, validated_params, signal, on_update)
    is_error = False
except Exception as e:
    result = ToolResult(content=[{"type": "text", "text": str(e)}])
    is_error = True
```

`is_error` lives on the **tool result message in the conversation** (set by the loop),
not on `ToolResult` itself. One path, one source of truth. The LLM sees the error
text and can react (retry, try a different approach, etc.).

**LLM error** → loop STOPS immediately:
```python
if assistant_msg.stop_reason in ("error", "aborted"):
    stream.push({"type": "turn_end", ...})
    stream.push({"type": "agent_end", ...})
    stream.end()
    return
```

### Error Recovery

After an error, consumer can call `agent.continue_run()` to retry from the same context.
No automatic retry — the consumer decides.

### Synthetic Error Messages

On unhandled exceptions, the Agent class creates a synthetic assistant message:
```python
error_msg = {
    "role": "assistant",
    "content": "",
    "stop_reason": "aborted" if signal.is_set() else "error",
    "error_message": str(exception),
    "usage": {"input": 0, "output": 0, ...},  # zeroed out
}
# Appended to message history so consumer can see what happened
```

### State Cleanup (always runs — finally block)

Regardless of success or failure:
```python
finally:
    self._state.is_streaming = False
    self._state.stream_message = None
    self._state.pending_tool_calls = set()
    self._signal = None  # agent-owned; fresh Event created on next prompt()/continue_run()
```

---

## Stop Reasons

```python
STOP_REASONS = {
    "stop":     # LLM finished naturally, no more tool calls
    "length":   # LLM hit max token limit
    "tool_use": # LLM wants to call tool(s) — loop continues
    "error":    # LLM request failed, stream error
    "aborted":  # User called abort()
}
```

- `"stop"` and `"length"` → agent ends normally
- `"tool_use"` → inner loop continues (execute tools, call LLM again)
- `"error"` and `"aborted"` → agent ends immediately

---

## Thinking / Reasoning

litellm returns two fields:

- `reasoning_content` (str) — universal, all providers
- `thinking_blocks` (list) — Anthropic only, includes cryptographic signatures

We store both. litellm handles sending the right format back to each provider.

```python
litellm.modify_params = True    # auto-handles thinking_blocks in tool call conversations
```

---

## Usage / Cost Tracking

Usage is tracked **per assistant message**, not aggregated by the loop.
See [Canonical Assistant Message Shape](#canonical-assistant-message-shape) for the exact usage fields.

Consumer aggregates across turns via `message_end` events or by summing from `agent.messages`.
Cost calculation is available from litellm's response metadata if needed — we pass through
whatever litellm provides rather than defining our own cost schema.

---

## Cancellation

Uses `asyncio.Event` as the cancellation signal (Python equivalent of pi's `AbortSignal`):

- Agent creates a fresh `asyncio.Event` for each `prompt()` / `continue_run()` call
- `agent.abort()` calls `signal.set()` — idempotent, safe to call multiple times
- Signal is passed through the entire chain: loop → LLM call → tool execute
- Tools check `signal.is_set()` to detect cancellation
- On cancel: current tool finishes or checks signal, loop emits agent_end
- Stop reason: `"aborted"` if `signal.is_set()`, otherwise `"error"`
- Signal cleared in finally block (new Event created on next run)

---

## What This Library Does NOT Include

- No tools (consumer provides them)
- No system prompt (consumer provides it)
- No UI (consumer builds it)
- No session persistence (consumer handles storage)
- No conversation compaction (consumer implements via transform_context hook)
- No MCP support
- No sub-agent orchestration (but the primitives support it — just call run_agent from a tool)
- No automatic retry (consumer implements via continue_run after error)
- No proxy/transport layer (optional future addition)

---

## References

- [pi-mono agent loop](https://github.com/badlogic/pi-mono/tree/main/packages/agent) — TypeScript source
- [Mario Zechner's blog post](https://mariozechner.at/posts/2025-11-30-pi-coding-agent/) — design philosophy
- [litellm docs](https://docs.litellm.ai/) — unified LLM interface
- [litellm reasoning/thinking](https://docs.litellm.ai/docs/reasoning_content) — thinking blocks handling
- [litellm streaming](https://docs.litellm.ai/docs/completion/stream) — async streaming

---

## Open Questions

### Decided
- ~~Dataclass vs Pydantic~~ → Pydantic for tool validation, dataclasses/dicts for messages (litellm compat)
- ~~kwargs vs dict for tool params~~ → dict (matches JSON Schema, `execute(tool_call_id, params, signal, on_update)`)
- ~~asyncio.Event vs Task.cancel()~~ → asyncio.Event as cancellation signal
- ~~jsonschema vs Pydantic~~ → Pydantic

### Still open
- [ ] EventStream: `asyncio.Queue` wrapper with `asyncio.Future` for result — or simpler?
- [ ] Events: typed dataclasses or plain dicts? Dicts are simpler for JSON serialization.
- [ ] Subscribe pattern: multiple listeners (pi's `subscribe()`) AND `async for`, or pick one?
- [ ] Abort propagation to litellm: mechanism TBD (spike during Phase 1), but the behavior contract is fixed — abort MUST result in stop_reason="aborted", cleanup MUST run, and the loop MUST emit agent_end.
- [ ] Testing: mock litellm or real API calls?
- [ ] Package name: `py_pi_agent`? `pi_agent`? `piloop`?
- [ ] `@tool` decorator for convenience? (Phase 4, but affects whether Pydantic model is required or optional)

---

## TODOs Before Building

### Phase 1: Core (must have)
- [ ] Implement `EventStream` (stream.py) — async queue + Future for result
- [ ] Implement types (types.py) — Tool, ToolResult, AgentConfig, AgentEvent, StopReason, AgentState
- [ ] Implement dual loop (loop.py) — streaming LLM response, tool execution, steering, follow-ups
- [ ] Implement Agent class (agent.py) — stateful wrapper with prompt/steer/follow_up/abort
- [ ] Tool argument validation with Pydantic + type coercion
- [ ] Thinking/reasoning: store reasoning_content + thinking_blocks, set litellm.modify_params
- [ ] Error handling: tool errors continue, LLM errors stop, synthetic error messages, finally cleanup
- [ ] Usage tracking: extract from litellm response, attach to assistant messages

### Phase 2: Prove it works
- [ ] Build a minimal CLI example (read user input, print streamed tokens)
- [ ] Build 2-3 toy tools (echo, sleep, fail) to test the full lifecycle
- [ ] Test steering: send interrupt mid-tool-execution, verify remaining tools skipped
- [ ] Test follow-up: queue message while agent is busy, verify it's delivered after idle
- [ ] Test abort: cancel mid-stream, verify cleanup and stop_reason="aborted"
- [ ] Test error recovery: trigger LLM error, call continue_run(), verify resumption
- [ ] Test multi-turn: verify loop handles multiple LLM↔tool rounds correctly

### Phase 3: Real-world integration
- [ ] Wire into existing agents/ FastHTML app (replace current sync loop)
- [ ] Add real tools: read_file, edit_file, bash, run_code (Modal sandbox)
- [ ] Add compaction via transform_context hook (chars/4 token estimation, backwards-walk)
- [ ] Add session persistence (save/load messages to disk or DB)
- [ ] SSE streaming to browser (tokens appear word-by-word)
- [ ] Usage/cost display in UI

### Phase 4: Nice to have (later)
- [ ] Default compaction utility (optional import, not required)
- [ ] Tool decorator for auto-generating JSON Schema from Python type hints
- [ ] Retry utility (wraps continue_run with backoff)
- [ ] Session ID support for provider-side caching
- [ ] Proxy/custom transport layer (streamFn equivalent)
- [ ] PyPI packaging and distribution

---

## Deliberately Excluded (from pi)

Features pi has that we intentionally skip, and why:

| Pi feature | Why excluded | Revisit? |
|---|---|---|
| Proxy/streamFn transport | Networking optimization, not core | Maybe later |
| Session ID | Provider-side caching, niche use case | Maybe later |
| TypeBox schema system | TS-specific, we use Pydantic (Python equivalent) | No |
| Declaration merging for custom messages | TS-specific pattern | No — Python has simpler extensibility |
| ThinkingBudgets (per-level token limits) | litellm handles this | Only if litellm doesn't cover it |
| Built-in provider implementations | litellm replaces all 6,800 lines | No |
| TUI / Web UI | Consumer's problem | No |
| Built-in tools | Consumer's problem | No |

See [COMPARISONS.md](COMPARISONS.md) for detailed comparisons with OpenAI Agents SDK and Anthropic Claude Agent SDK.
