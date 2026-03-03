# py-pi-agent — WIP Spec

A Python core agent loop library inspired by [pi-mono](https://github.com/badlogic/pi-mono)'s `packages/agent`. Framework-agnostic, plug-and-play, usable from FastHTML, FastAPI, Slack, CLI, or anything else.

## Dependencies

- `asyncio` — async runtime (stdlib)
- `litellm` — unified LLM interface (Chat Completions format for all providers)
- `pydantic` — tool argument validation + type coercion

## Target Models

These are the models we test against and must work with:

- `anthropic/claude-opus-4-6`
- `anthropic/claude-sonnet-4-6`
- `gemini/gemini-3-flash-preview`
- `gemini/gemini-3.1-pro-preview`
- `gpt-5.2`

All use litellm model strings. Tests should pass against all five.

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

## Key Design Decisions (and why)

**Why litellm?** Pi built ~6,800 lines of provider-specific code (Anthropic, OpenAI, Google, Bedrock,
etc.) to normalize streaming, tool calls, and thinking traces across providers. litellm does the same
thing as a maintained Python library. One `litellm.acompletion()` call replaces all of that.

**Provider boundary principle:** `loop.py` is provider-agnostic but boundary-defensive. No
provider-specific control flow (`if anthropic:`, `if gemini:`) — but two kinds of leakage are
acceptable at the litellm boundary:

1. **Defensive reads** when litellm exposes slightly different shapes per provider (e.g.,
   `_extract_usage` checks both Anthropic's top-level cache fields and OpenAI's nested ones).
2. **Opaque preservation** of fields litellm gives us that we don't interpret (e.g.,
   `provider_specific_fields`, `thinking_blocks`, `reasoning_content`).

If you find yourself interpreting provider-specific semantics in the loop, that's the smell.
Push request-shape differences into `convert_to_llm` — that's the caller's responsibility.

**Why Pydantic (not jsonschema)?** Pi uses AJV (JavaScript JSON Schema validator) with `coerceTypes: true`
to fix LLM mistakes like sending `"42"` instead of `42`. Python's `jsonschema` library doesn't coerce.
Pydantic does — it validates AND coerces naturally. It's the Python equivalent of AJV + TypeBox.

**Why asyncio (not threading or trio)?** litellm, FastHTML, FastAPI, and Textual all use asyncio.
Threading can't do steering/interruption cleanly. Trio is cleaner but fights the ecosystem. asyncio
is the practical choice — everything we want to plug into already speaks it.

**Why two layers (loop functions + Agent class)?** Same as pi. The raw loop (`run_loop`, `agent_loop`,
`agent_loop_continue`) is the engine — stateless, returns an EventStream. The Agent class wraps it
with state management (messages, queues, streaming flag, abort). Consumers can use either layer.
Most will use the Agent class. Advanced consumers may use the raw loop directly.

**Why sequential tool execution (not parallel)?** Steering. After each tool finishes, the loop checks
for user interruption. If tools run in parallel, you can't skip the rest — you'd have to cancel
already-running tools. Sequential is simpler and matches pi.

**How sub-agents work:** The core loop has no sub-agent concept. A tool can internally call `run_agent()`
to spawn a child agent — it's just a function calling another function. The parent sees it as a slow
tool that returned text. No special machinery needed. The loop's primitives (EventStream, tools, config)
are sufficient.

---

## EventStream

Async producer-consumer queue. The loop pushes events, consumers iterate.

```python
class EventStream:
    """Async event stream with producer-consumer pattern."""

    def push(self, event: dict):
        """Producer pushes events (non-blocking)."""

    def end(self, result):
        """Signal stream completion and store final result."""

    async def result(self) -> list:
        """Await final result (all messages)."""

    async def __aiter__(self):
        """Consumer iterates: async for event in stream: ..."""
```

Built on `asyncio.Queue`. Events buffer in an unbounded queue if consumer is slow.

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
        if event["delta_type"] == "text_delta":
            console.print(event["delta"]["content"], end="")
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
           (skip initial steering poll in run_loop to avoid double-check)
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
    # All setters blocked while is_streaming. Changes take effect on next prompt()/continue_run().
    # set_tools enables dynamic capabilities: permission escalation, plan-mode tool switching,
    # preset configurations. Pi's coding agent uses this extensively with a tool registry pattern
    # (all discovered tools + active subset). We provide the primitive; consumers build on it.

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
                                    # "off" = don't send reasoning_effort to litellm
                                    # others passed to litellm's reasoning_effort param directly
                                    # litellm maps these to provider-specific budgets internally
                                    # (e.g., Anthropic: "low" → budget_tokens=1024)
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

### Internal `_run_loop()` — how the Agent wires into the loop

The Agent's internal method that connects everything. Same pattern as pi's `_runLoop()` in `agent.ts`.

```python
async def _run_loop(self, prompts=None):
    """Internal: create signal, build config, call loop, iterate events to update state."""
    self._signal = asyncio.Event()  # fresh cancellation signal per run
    self._state.is_streaming = True
    self._state.error = None
    self._running_future = asyncio.get_running_loop().create_future()  # for wait_for_idle()

    try:
        # Build AgentConfig — wire queues as hooks
        config = AgentConfig(
            model=self._state.model,
            convert_to_llm=self._convert_to_llm,
            transform_context=self._transform_context,
            get_steering_messages=self._dequeue_steering,   # returns list or None
            get_follow_up_messages=self._dequeue_follow_ups,
            reasoning_effort=None if self._state.thinking_level == "off"
                             else self._state.thinking_level,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            num_retries=self._num_retries,
        )

        # Snapshot context
        context = AgentContext(
            system_prompt=self._state.system_prompt,
            messages=list(self._state.messages),  # shallow copy — loop appends to its own copy
            tools=list(self._state.tools) if self._state.tools else None,
        )

        # Call the appropriate loop entry point
        if prompts is not None:
            stream = agent_loop(prompts, context, config, self._signal)
        else:
            stream = agent_loop_continue(context, config, self._signal)

        # Iterate events — update state + publish to subscribers
        async for event in stream:
            self._handle_event(event)
            self._emit(event)

        # After stream ends, update messages from result
        new_messages = await stream.result()
        self._state.messages.extend(new_messages)

    except Exception as e:
        # Unhandled error — create synthetic error message
        error_msg = { ... }  # see Synthetic Error Messages section
        self._state.messages.append(error_msg)
        self._state.error = str(e)

    finally:
        self._state.is_streaming = False
        self._state.stream_message = None
        self._state.pending_tool_calls = set()
        self._signal = None
        if not self._running_future.done():
            self._running_future.set_result(None)  # unblock wait_for_idle()
```

### Event handling — updating state from events

`_handle_event()` keeps `AgentState` in sync with the running loop:

```python
def _handle_event(self, event):
    t = event["type"]
    if t == "message_start":
        if event["message"].get("role") == "assistant":
            self._state.stream_message = event["message"]
    elif t == "message_update":
        self._state.stream_message = event["message"]
    elif t == "message_end":
        self._state.stream_message = None
    elif t == "tool_execution_start":
        self._state.pending_tool_calls.add(event["tool_call_id"])
    elif t == "tool_execution_end":
        self._state.pending_tool_calls.discard(event["tool_call_id"])
```

### Queue dequeuing — respects mode

```python
def _dequeue_steering(self):
    if not self._steering_queue:
        return None
    if self.steering_mode == "all":
        msgs = list(self._steering_queue)
        self._steering_queue.clear()
        return msgs
    else:  # one-at-a-time
        return [self._steering_queue.pop(0)]

# Same pattern for _dequeue_follow_ups
```

### Partial message handling on abort

Pi carefully checks whether a partial message has real content before appending.
If the agent is aborted mid-stream, the stream may end with a partial message that has
only whitespace or empty content. Skip these — don't pollute the message history.

### `convert_to_llm` default

The Agent class provides a default `convert_to_llm` that filters to standard roles:
```python
def _default_convert_to_llm(messages):
    return [m for m in messages if m.get("role") in ("user", "assistant", "tool")]
```
Consumers override this for custom message types or provider-specific workarounds
(e.g., OpenAI multimodal tool results).

---

## Event Types

> **Naming convention**: Pi uses camelCase (`toolUse`, `toolcall_delta`). We use snake_case
> (`tool_use`, `tool_call_delta`) per Python convention. Same semantics, different casing.

```python
# Agent lifecycle
{"type": "agent_start"}
{"type": "agent_end", "messages": [...]}  # only NEW messages from this run, not full history

# Turn lifecycle (one LLM call + tool executions)
{"type": "turn_start"}
{"type": "turn_end", "message": ..., "tool_results": [...]}

# Message lifecycle — emitted for ALL message types (user, assistant, tool result, steering)
{"type": "message_start", "message": ...}
{"type": "message_update", "message": ..., "delta": ..., "delta_type": ...}  # assistant only — streaming
{"type": "message_end", "message": ...}

# Streaming deltas (within message_update)
#
# The "delta" field is a plain JSON-serializable dict with only the fields
# present in this chunk (no litellm objects leak through):
#   {"content": "token"}                — text delta
#   {"reasoning_content": "token"}      — thinking/reasoning delta
#   {"thinking_blocks": [...]}          — Anthropic thinking blocks (includes signatures)
#   {"tool_calls": [{"index": 0, ...}]} — tool call fragments as dicts
#
# "delta_type" indicates which kind:
#   "text_delta"         — delta has "content"
#   "thinking_delta"     — delta has "reasoning_content" or "thinking_blocks"
#   "tool_call_delta"    — delta has "tool_calls"
#
# Note: deltas are for UI display only. Tool execution uses the finalized
# assistant message from litellm, not reassembled deltas.

# Tool execution lifecycle
{"type": "tool_execution_start", "tool_call_id": ..., "tool_name": ..., "args": ...}
{"type": "tool_execution_update", "tool_call_id": ..., "tool_name": ..., "args": ..., "partial": ...}
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
    parameters: dict        # JSON Schema for arguments
                            # Sent to litellm as: {"type": "function", "function": {"name", "description", "parameters"}}
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
- On validation failure: error becomes a tool result message with `is_error=True`,
  sent back to the LLM. The LLM sees the error and can retry with corrected args.
  No retry logic in the loop — the LLM is the retry mechanism.
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
- If steering: skip remaining tools, generate synthetic tool result messages for each skipped tool
  with `is_error=True` and content "Skipped due to queued user message." — this keeps tool call/result
  pairing balanced (some providers reject orphaned tool calls without matching results)
- Tools receive `signal` for cancellation and `on_update` callback for streaming

---

## Dual Loop Architecture

> **Note:** Pseudocode below omits optional-hook guards (`if config.hook:`) and
> `maybe_async()` wrappers for readability. The implementation will handle these.

```python
# Entry points — emit agent_start, then delegate to run_loop
# (In pi-mono these are agentLoop() and agentLoopContinue())

def agent_loop(messages, context, config, signal) -> EventStream:
    stream = EventStream()
    async def run():
        new_messages = list(messages)
        local_ctx = AgentContext(                         # shallow copy
            system_prompt=context.system_prompt,
            messages=list(context.messages) + list(messages),
            tools=context.tools,
        )
        stream.push({"type": "agent_start"})
        stream.push({"type": "turn_start"})
        for m in messages:
            stream.push({"type": "message_start", "message": m})
            stream.push({"type": "message_end", "message": m})
        await run_loop(local_ctx, new_messages, config, signal, stream)
    asyncio.create_task(run())
    return stream

# The engine — handles normal + error termination (agent_end + stream.end).
# Entry points have a finally fallback for CancelledError / unexpected BaseExceptions.
# Same as pi-mono's runLoop (agent-loop.ts:104-198).
async def run_loop(context, new_messages, config, signal, stream):
    pending_messages = await config.get_steering_messages()
    first_turn = True

    # OUTER LOOP: handles follow-up messages after agent settles
    while True:
        has_tool_calls = True

        # INNER LOOP: LLM calls + tool execution + steering
        while has_tool_calls or pending_messages:
            if not first_turn:
                stream.push({"type": "turn_start"})  # skip on first — already emitted by entry point
            first_turn = False

            # 1. Inject pending messages into context
            # 2. Stream LLM response → returns assistant_msg, appended to context + new_messages
            # 3. If error/aborted: emit turn_end + agent_end + stream.end, return
            # 4. Execute tools sequentially (check steering after each)
            #    - execute_tool_calls returns tool_results AND any steering messages found
            #    - Skipped tools get synthetic error results to keep tool call/result pairing balanced
            # 5. Push turn_end
            # 6. Use steering from execute_tool_calls first, else poll config.get_steering_messages

        # Agent would stop — check for queued follow-ups
        follow_ups = await config.get_follow_up_messages()
        if follow_ups:
            pending_messages = follow_ups
            continue

        break

    # Normal exit
    stream.push({"type": "agent_end", "messages": new_messages})
    stream.end(new_messages)
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
    messages = context.messages
    if config.transform_context:
        messages = await config.transform_context(messages, signal)

    # Hook: convert to LLM format (filter out UI-only message types)
    llm_messages = config.convert_to_llm(messages)

    # Build tool schemas from context.tools for litellm
    tool_schemas = [
        {"type": "function", "function": {"name": t.name, "description": t.description, "parameters": t.parameters}}
        for t in (context.tools or [])
    ] or None  # None if no tools (don't send empty list)

    response = await litellm.acompletion(
        model=config.model,
        messages=llm_messages,
        tools=tool_schemas,
        stream=True,
        stream_options={"include_usage": True},  # required to get usage on final streaming chunk
        reasoning_effort=config.reasoning_effort,
    )

    # Stream 3 types of chunks
    async for chunk in response:
        delta = chunk.choices[0].delta

        if delta.reasoning_content:     # thinking tokens
            stream.push({"type": "message_update", "message": partial_msg,
                         "delta": delta, "delta_type": "thinking_delta"})

        if delta.content:               # text tokens
            stream.push({"type": "message_update", "message": partial_msg,
                         "delta": delta, "delta_type": "text_delta"})

        if delta.tool_calls:            # tool call fragments
            stream.push({"type": "message_update", "message": partial_msg,
                         "delta": delta, "delta_type": "tool_call_delta"})

    # Build finalized assistant message via litellm.stream_chunk_builder().
    # Chunks collected during streaming → stream_chunk_builder assembles the
    # complete ModelResponse (content, tool_calls, thinking_blocks, usage, etc.)
    # This is Pi's equivalent of await response.result().
    final = litellm.stream_chunk_builder(chunks)
    assistant_msg = build_assistant_message(final)
    context.messages.append(assistant_msg)
```

### Canonical Assistant Message Shape

Built from `litellm.stream_chunk_builder(chunks)` — the finalized `ModelResponse`
assembled from collected streaming chunks (equivalent to Pi's `response.result()`):

```python
{
    "role": "assistant",
    "content": "response text" | None,              # None when only tool calls
    "tool_calls": [                                  # from litellm response
        {"id": "call_123", "type": "function",
         "function": {"name": "bash", "arguments": '{"command": "ls"}'},
         "provider_specific_fields": {...} | None},  # Gemini thought_signatures on tool calls
    ] | None,
    "thinking_blocks": [...] | None,                 # Anthropic only, includes signatures
    "reasoning_content": "thinking text" | None,     # universal, all providers
    "provider_specific_fields": {...} | None,        # opaque bag from litellm (Gemini thought_signatures, etc.)
    "usage": {                                       # from litellm response.usage
        "prompt_tokens": 1234,                       # litellm field name (input tokens)
        "completion_tokens": 567,                    # litellm field name (output tokens)
        "total_tokens": 1801,                        # litellm field name
        "cache_read_tokens": 890,                    # from usage.prompt_tokens_details.cached_tokens
        "cache_creation_tokens": 0,                  # from usage.prompt_tokens_details.cache_creation_tokens
    },
    "stop_reason": "stop" | "tool_calls" | "length" | "error" | "aborted",
    "timestamp": 1708531200000,                      # Unix ms — added to every message (same as pi)
}
```

This is our internal message format, built from the finalized `ModelResponse` returned by
`litellm.stream_chunk_builder(chunks)`. During streaming, chunks are collected in a list.
After streaming completes, `stream_chunk_builder` assembles the complete response:
- `content`, `tool_calls`, `reasoning_content`, `thinking_blocks` come from `final.choices[0].message`
- `provider_specific_fields` comes from `final.choices[0].message` — opaque, preserved without interpretation
- `stop_reason` is captured from the raw chunks during streaming (not from `stream_chunk_builder`,
  which has a bug: the usage-only chunk from `stream_options={"include_usage": True}` overwrites the
  real `finish_reason` with `None`). Mapped by litellm: Anthropic "tool_use" → "tool_calls",
  "end_turn" → "stop". Or "aborted"/"error" set by our loop.
- `usage` comes from `final.usage` (requires `stream_options={"include_usage": True}` on the original call)

The loop reads `tool_calls` to decide whether to continue. Thinking fields are preserved
for future messages (Anthropic requires them back).

**Important:** `convert_to_llm` must pass assistant messages through with all fields intact —
including `thinking_blocks` and `reasoning_content`. Anthropic requires thinking blocks from
previous turns to be sent back in subsequent requests. Stripping them will cause errors
when extended thinking is enabled. The `convert_to_llm` hook filters/transforms *non-standard*
message types (custom, UI-only); standard `user`, `assistant`, and `tool` messages pass through.

---

## AgentContext (loop state)

The data snapshot the loop operates on. Created once at the start of each run.
Same as pi's `AgentContext` — separates **state** (what the loop works with)
from **behavior** (how the loop works, i.e. config/hooks).

```python
@dataclass
class AgentContext:
    system_prompt: str                  # system prompt for LLM — loop reads, never mutates
    messages: list                      # conversation history — loop APPENDS new messages here
    tools: list[Tool] | None = None    # available tools — loop reads, never mutates
```

`system_prompt` and `tools` are read-only during the run. `messages` is the one field the
loop mutates — it appends assistant messages and tool result messages as the run progresses.

The Agent class snapshots its current state into a context at the start of each `prompt()` /
`continue_run()` call. The loop then works exclusively with this snapshot. Even if
`agent.set_tools()` is called externally, the running loop won't see it.

---

## AgentConfig (loop behavior)

```python
@dataclass
class AgentConfig:
    model: str                              # litellm model string

    # REQUIRED: filter/transform messages for LLM (can be sync or async — loop awaits either)
    # Must pass user/assistant/tool messages through intact (including thinking_blocks).
    # Used to convert custom message types (bashExecution, summaries) to user messages
    # and filter out UI-only types.
    # Typically sync (just filtering a list). For simple agents: lambda msgs: msgs
    # Provider gotcha: OpenAI tool messages only support string content — images are
    # silently dropped. A provider-aware convert_to_llm can strip images from tool
    # results and re-inject them as synthetic user messages (same workaround as pi).
    convert_to_llm: Callable

    # OPTIONAL hooks (all can be sync or async — loop awaits either)
    transform_context: Callable = None      # compaction, pruning, injection (typically async — may call LLM)
    get_steering_messages: Callable = None  # check for user interruption (typically async)
    get_follow_up_messages: Callable = None # check for queued messages (typically async)

    # LLM parameters
    reasoning_effort: str = None            # "minimal", "low", "medium", "high", "xhigh"
                                            # None = don't send to litellm (no thinking)
                                            # litellm maps these to provider-specific budgets internally
                                            # (Anthropic → budget_tokens, Gemini → thinkingBudget, etc.)
                                            # No need for explicit ThinkingBudgets — litellm handles it
    max_tokens: int = None
    temperature: float = None

    # Retry behavior — passed to litellm, not implemented by the loop
    num_retries: int = None
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

**Provider support:** Only Anthropic has a native `is_error` field on tool results.
OpenAI and Gemini don't — litellm strips it during conversion. The error *text* in
content is what makes all providers understand the tool failed; `is_error` is a bonus
signal for Anthropic and for our own event system / consumer message inspection.

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
    "content": None,
    "stop_reason": "aborted" if signal.is_set() else "error",
    "error_message": str(exception),
    "usage": {},  # zeroed out
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
    "stop":       # LLM finished naturally, no more tool calls
    "length":     # LLM hit max token limit
    "tool_calls": # LLM wants to call tool(s) — loop continues
    "error":      # LLM request failed, stream error
    "aborted":    # User called abort()
}
```

> **Note:** litellm normalizes all providers to OpenAI format. Anthropic's `"tool_use"` and
> `"end_turn"` become `"tool_calls"` and `"stop"` respectively (see `map_finish_reason` in litellm).

- `"stop"` and `"length"` → agent ends normally
- `"tool_calls"` → inner loop continues (execute tools, call LLM again)
- `"error"` and `"aborted"` → agent ends immediately

---

## Thinking / Reasoning

litellm returns two fields:

- `reasoning_content` (str) — universal, all providers
- `thinking_blocks` (list) — Anthropic only, includes cryptographic signatures

We store both. litellm handles sending the right format back to each provider.

### Thinking block assembly

`litellm.stream_chunk_builder(chunks)` handles merging thinking block fragments
(partial text + signature finalization) into complete blocks on the finalized
`ModelResponse`. The loop does not need to merge thinking blocks manually — it
collects chunks during streaming and delegates assembly to `stream_chunk_builder`.

```python
litellm.modify_params = True
```

`modify_params` is a litellm global that auto-fixes common message format issues:
- Inserts placeholder user messages when providers require alternating roles (Anthropic, Bedrock)
- Adds a dummy tool definition when messages contain tool_call blocks but no `tools=` param
- Drops the `thinking` param when assistant messages lack thinking_blocks (prevents Anthropic errors)
- Sanitizes orphaned tool calls/results (missing tool_result for tool_use, etc.)

We need this because our loop may produce message sequences that trip provider-specific
constraints (e.g., consecutive user messages, tool calls without tools param on retry).

---

## Usage / Cost Tracking

Usage is tracked **per assistant message**, not aggregated by the loop.
See [Canonical Assistant Message Shape](#canonical-assistant-message-shape) for the exact usage fields.

**Streaming usage:** By default, litellm streaming does NOT include usage on chunks. You must pass
`stream_options={"include_usage": True}` to `acompletion()` to get a usage object on the final chunk.
Without this, `chunk.usage` is None on every chunk. We always pass this option.

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

## Resolved Questions

All decided:

- ~~Dataclass vs Pydantic~~ → Pydantic for tool validation, dataclasses/dicts for messages (litellm compat)
- ~~kwargs vs dict for tool params~~ → dict (matches JSON Schema, `execute(tool_call_id, params, signal, on_update)`)
- ~~asyncio.Event vs Task.cancel()~~ → asyncio.Event as cancellation signal
- ~~jsonschema vs Pydantic~~ → Pydantic
- ~~EventStream implementation~~ → `asyncio.Queue` for event buffer + `asyncio.Future` for final result + `__aiter__` for consumers. Matches pi's pattern (queue + waiting callbacks + Promise).
- ~~Events format~~ → Plain dicts. Match litellm's format, trivial JSON serialization, no parallel type system. Same as pi (plain objects).
- ~~Subscribe pattern~~ → Both `async for` AND `subscribe()`. Same as pi. `async for` is the primary consumer API. `subscribe()` is used by the Agent class internally to update state from events.
- ~~Abort propagation~~ → Check `signal.is_set()` between streaming chunks from litellm. If set, break out of the chunk loop, build a partial message with `stop_reason="aborted"`. For tool execution, tools already receive the signal. Behavior contract: abort → stop_reason="aborted", cleanup runs, agent_end emitted.
- ~~Testing strategy~~ → Real API calls for integration tests (Phase 2). Unit tests for EventStream/types need no LLM. Mock only for error paths hard to trigger with real APIs.
- ~~Package name~~ → `py_pi_agent` (directory) / `py-pi-agent` (package). Already in pyproject.toml.
- ~~@tool decorator~~ → Phase 4. `params_model` stays optional — tools can provide just JSON Schema (no Pydantic model required). Decorator will auto-generate both from type hints later.

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

### Phase 2: Test runner + live tests
- [ ] Build test runner (see Test Runner section below)
- [ ] Build test tools covering all features (see Test Tools below)
- [ ] All tests use real API calls (target models below)
- [ ] Verify: streaming tokens arrive word-by-word
- [ ] Verify: tool calls execute and results flow back to LLM
- [ ] Verify: multimodal — tool returns image, LLM sees it, references it in follow-up
- [ ] Verify: thinking/reasoning traces stored and visible in events
- [ ] Verify: multi-turn — conversation context carries forward across tool rounds
- [ ] Verify: steering — interrupt mid-tool-execution, remaining tools skipped
- [ ] Verify: follow-up — queued message delivered after agent idles
- [ ] Verify: abort — cancel mid-stream, cleanup runs, stop_reason="aborted"
- [ ] Verify: error recovery — tool throws, LLM sees error and adapts
- [ ] Verify: continue_run() — resume after error, assistant-tail edge case
- [ ] Verify: on_update — tool streams partial results, events arrive in real time
- [ ] Verify: usage — token counts present on assistant messages

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
| ThinkingBudgets (per-level token limits) | litellm maps reasoning_effort → provider budgets internally (Anthropic: budget_tokens, Gemini: thinkingBudget) | No — confirmed litellm handles it |
| getApiKey hook | Dynamic API key resolution per LLM call (e.g., expiring OAuth tokens). litellm has its own key management for now | Maybe later if OAuth rotation needed |
| Built-in provider implementations | litellm replaces all 6,800 lines | No |
| TUI / Web UI | Consumer's problem | No |
| Built-in tools | Consumer's problem | No |
| Cross-model thinking conversion | Switching models mid-conversation leaves orphaned thinking blocks; `transform_context` hook can strip them if needed | Only if it causes real issues |

See [COMPARISONS.md](COMPARISONS.md) for detailed comparisons with OpenAI Agents SDK and Anthropic Claude Agent SDK.

---

## Test Runner

A stripped-down interactive agent — the first real consumer of py-pi-agent. Inspired by
pi's coding-agent but minimal. Runs in the terminal, exercises every feature of the core loop.

### Structure

```
tests/
    runner.py           # Interactive CLI runner
    tools.py            # Test tools covering all features
    system_prompt.py    # System prompt for test agent
```

### Runner behavior

```bash
$ uv run python tests/runner.py --model anthropic/claude-sonnet-4-6
Tools: echo, bash, read_file, write_file, generate_chart, analyze_image,
       slow_task, always_fail
Type a message (ctrl+c to abort mid-run, /steer to interrupt, /quit to exit)

You: make me a bar chart of Q1 through Q4 sales
[thinking] Let me generate that chart...
[tool_start] generate_chart {"title": "Sales by Quarter", "data": [10,25,18,30]}
[tool_update] rendering...
[tool_end] ok (0.3s)
[image: data:image/png;base64,iVBOR... (24KB)]
Assistant: Here's your quarterly sales chart. Q4 was the strongest...

You: what does the chart show?
Assistant: The chart I just generated shows four quarters...
(^ proves LLM saw the image in conversation history)

You: /steer actually make it a pie chart instead
[steering] Redirecting...
```

### What the runner prints for each event type

| Event | Display |
|---|---|
| `message_update` (text_delta) | Print token text inline (streaming) |
| `message_update` (thinking_delta) | Print dim/gray `[thinking] ...` |
| `tool_execution_start` | `[tool_start] name {args}` |
| `tool_execution_update` | `[tool_update] partial text` |
| `tool_execution_end` | `[tool_end] ok/error (duration)` |
| `message_end` (with images) | `[image: type, size]` |
| `turn_end` | Separator line |
| `agent_end` | Done, show usage summary |

### Test Tools

All tools are real (no mocks) and use live API calls where applicable.

```python
# 1. echo — simplest possible tool, validates basic tool calling
async def echo(tool_call_id, params, signal=None, on_update=None):
    """Echo back a message. Params: {"message": str}"""
    return ToolResult(content=[{"type": "text", "text": params["message"]}])

# 2. bash — real subprocess, tests on_update streaming
async def bash(tool_call_id, params, signal=None, on_update=None):
    """Run a shell command. Params: {"command": str}"""
    process = await asyncio.create_subprocess_shell(
        params["command"], stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    output = ""
    async for line in process.stdout:
        text = line.decode()
        output += text
        if on_update:
            on_update(ToolResult(content=[{"type": "text", "text": output}]))
    stderr = (await process.stderr.read()).decode()
    if stderr:
        output += f"\nstderr: {stderr}"
    return ToolResult(content=[{"type": "text", "text": output}])

# 3. read_file — local filesystem, tests simple I/O
async def read_file(tool_call_id, params, signal=None, on_update=None):
    """Read a file. Params: {"path": str}"""
    content = open(params["path"]).read()
    return ToolResult(content=[{"type": "text", "text": content}])

# 4. write_file — local filesystem
async def write_file(tool_call_id, params, signal=None, on_update=None):
    """Write a file. Params: {"path": str, "content": str}"""
    from pathlib import Path
    Path(params["path"]).parent.mkdir(parents=True, exist_ok=True)
    Path(params["path"]).write_text(params["content"])
    return ToolResult(content=[{"type": "text", "text": f"Wrote {params['path']}"}])

# 5. generate_chart — MULTIMODAL: returns text + image
async def generate_chart(tool_call_id, params, signal=None, on_update=None):
    """Generate a matplotlib chart. Params: {"title": str, "labels": [str], "values": [num]}"""
    import matplotlib.pyplot as plt
    import io, base64
    fig, ax = plt.subplots()
    ax.bar(params.get("labels", ["A","B","C"]), params.get("values", [3,7,2]))
    ax.set_title(params.get("title", "Chart"))
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    img_b64 = base64.b64encode(buf.getvalue()).decode()
    return ToolResult(
        content=[
            {"type": "text", "text": f"Generated: {params.get('title', 'Chart')}"},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
        ],
        details={"chart_type": "bar"},  # UI-only metadata
    )

# 6. slow_task — tests cancellation + on_update streaming
async def slow_task(tool_call_id, params, signal=None, on_update=None):
    """Slow countdown. Params: {"seconds": int}"""
    for i in range(params.get("seconds", 5), 0, -1):
        if signal and signal.is_set():
            raise Exception("Aborted")
        if on_update:
            on_update(ToolResult(content=[{"type": "text", "text": f"{i}..."}]))
        await asyncio.sleep(1)
    return ToolResult(content=[{"type": "text", "text": "Done!"}])

# 7. always_fail — tests error path
async def always_fail(tool_call_id, params, signal=None, on_update=None):
    """Always raises an error. Params: {"message": str}"""
    raise Exception(params.get("message", "This tool always fails"))
```

### System prompt (for test runner)

```
You are a test agent for the py-pi-agent library. You have these tools:

- echo: Echo back a message. Use for simple tests.
- bash: Run shell commands. Output streams line by line.
- read_file: Read a local file.
- write_file: Write content to a local file.
- generate_chart: Generate a matplotlib bar chart. Returns an image.
- slow_task: Count down for N seconds. Use to test cancellation.
- always_fail: Always throws an error. Use to test error handling.

When asked to create charts or visualizations, use generate_chart.
When asked to read or write files, use the file tools.
When asked to run commands, use bash.
If the user asks you to do something that will fail, use always_fail.
Always explain what you're doing and describe any images you receive.
```

### What this tests end-to-end

| Conversation | Features exercised |
|---|---|
| "echo hello world" | Basic tool call, text result |
| "run `ls -la`" | bash, on_update streaming |
| "read the SPEC.md file" | read_file, large text result |
| "make me a chart of A=3 B=7 C=2" | generate_chart, multimodal response (text + image) |
| "what does that chart show?" | Multi-turn, LLM references previous image |
| "count down from 5" then ctrl+c | slow_task, abort mid-tool, on_update |
| "count down from 10" then `/steer stop and echo done instead" | Steering, skip remaining |
| "try to fail with message 'test error'" | always_fail, error handling, LLM reacts to error |
| "write a file then read it back" | Multi-tool round, write_file + read_file |
| (use reasoning model) "explain quantum computing" | Thinking traces in events |
