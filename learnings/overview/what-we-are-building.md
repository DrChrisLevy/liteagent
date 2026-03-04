# What We're Building

## Introduction

[Pi](https://github.com/badlogic/pi-mono) is a coding agent by [Mario Zechner](https://mariozechner.at/).
I'm not building the coding agent. I'm learning and building from it's core agent loop.

At its heart, every LLM agent does the same thing:

```
    ┌──────────────────────────────────┐
    │  User sends a message            │
    └──────────────┬───────────────────┘
                   │
                   v
    ┌──────────────────────────────────┐
    │  Send messages to LLM            │◄───────────┐
    └──────────────┬───────────────────┘            │
                   │                                │
                   v                                │
            ┌─────────────┐                         │
            │ Tool calls? │── no ──► Done           │
            └──────┬──────┘                         │
                   │ yes                            │
                   v                                │
    ┌──────────────────────────────────┐            │
    │  Execute tools, collect results  │────────────┘
    └──────────────────────────────────┘
```

LLM responds → if it wants to call tools, run them, feed results back, repeat.
If it doesn't want tools, it's done. That's the whole thing.

The differences between frameworks are about what happens *around* this loop:
How do you stream tokens? How do you interrupt a running agent? How do you
manage conversation history? How do you handle errors?

Pi's answer in its core agent loop is a **dual while-loop**.

## The Dual Loop

Pi doesn't have a flat loop. It has two nested loops with different jobs:

```
OUTER LOOP ─── "Should the agent wake up again?"
│
│   INNER LOOP ─── "Keep calling LLM + tools until done"
│   │
│   │   1. Any steering messages? (user interrupted)
│   │      → Inject them into context
│   │
│   │   2. Call LLM (streaming tokens to consumer)
│   │
│   │   3. Tool calls in response?
│   │      → Execute them ONE AT A TIME
│   │      → After EACH tool: check steering queue
│   │      → If user interrupted: skip remaining tools
│   │
│   │   4. More tool calls or steering? → loop back to 1
│   │
│   └── Inner loop done (LLM finished, no more tools)
│
│   Any follow-up messages queued?
│   → yes: feed them in, go back to inner loop
│   → no: we're done
│
└── Agent stops
```

### Why two loops?

**Steering** and **follow-ups** solve different problems:

- **Steering** = "Stop what you're doing, do this instead." Immediate. The user
  types a correction while the agent is executing tools. The inner loop checks
  for steering after each tool, skips the rest, and feeds the new instruction
  to the LLM.

- **Follow-up** = "When you're done, also do this." Deferred. The user queues a
  message while the agent is busy. The outer loop delivers it after the agent
  finishes its current work.


### Why sequential tool execution?

The LLM might request 5 tool calls at once. Pi runs them **one at a time**,
not in parallel. This is deliberate:

```
Parallel execution:        Sequential execution:

  tool 1 ──────►             tool 1 ──► check steering ──►
  tool 2 ──────►             tool 2 ──► check steering ──►
  tool 3 ──────►             tool 3 ──► check steering ──►
  tool 4 ──────►             (user interrupts after tool 2)
  tool 5 ──────►             tool 4 ──► SKIPPED
                             tool 5 ──► SKIPPED
  (can't interrupt)
                             User's new message injected ──► LLM
```

Sequential execution enables steering. After each tool, the loop can check:
"Did the user say something?" If yes, skip the rest.

## What We're Building

**py-pi-agent** is a Python port of pi's core agent loop. Same architecture,
Python idioms, one major shortcut: we use [litellm](https://docs.litellm.ai/)
instead of hand-rolling LLM provider code.

Pi's `packages/ai/` directory contains thousands of lines of provider-specific
streaming code. Per-provider message conversion, chunk parsing, error handling,
retry logic. All for Anthropic, OpenAI, Google, etc.

We replace all of that with one function call:

```python
response = await litellm.acompletion(
    model="anthropic/claude-sonnet-4-6",  # or "gpt-5.2", "gemini/...", etc.
    messages=messages,
    tools=tool_schemas,
    stream=True,
)
```

litellm normalizes every provider to the OpenAI Chat Completions format.
Streaming chunks, tool calls, thinking tokens, usage tracking (it's all unified).


### The four files

```
py_pi_agent/
    stream.py     EventStream — async queue that connects producer to consumer
    types.py      Tool, ToolResult, AgentConfig — the vocabulary
    loop.py       The dual while-loop — the engine
    agent.py      Agent class — stateful wrapper with prompt/steer/abort
```

No tools, no system prompt, no UI, no persistence.
All of that is the consumer's problem.

### The layers

```
┌─────────────────────────────────────────────────────┐
│  YOUR APP                                           │
│  (CLI, FastHTML, FastAPI, Slack bot, whatever)       │
│                                                     │
│  - Defines tools (read_file, bash, run_code, etc.)  │
│  - Defines system prompt                            │
│  - Handles UI / transport                           │
│  - Manages persistence                              │
├─────────────────────────────────────────────────────┤
│  py-pi-agent                                        │
│                                                     │
│  Agent class ─── prompt(), steer(), follow_up()     │
│       │                                             │
│  Loop ─────────── dual while-loop, tool execution   │
│       │                                             │
│  EventStream ──── async producer-consumer queue     │
├─────────────────────────────────────────────────────┤
│  litellm                                            │
│                                                     │
│  - Unified LLM interface (100+ providers)           │
│  - Streaming, tool calls, thinking tokens           │
│  - Usage tracking, error normalization              │
└─────────────────────────────────────────────────────┘
```

### How it flows

Here's a concrete example: user asks the agent to read a file.

```
 User                     Agent                    LLM (via litellm)
  │                         │                         │
  │  prompt("read main.py") │                         │
  │ ───────────────────────►│                         │
  │                         │                         │
  │   ◄─ agent_start        │  messages + tools       │
  │   ◄─ turn_start         │ ───────────────────────►│
  │   ◄─ message_start      │                         │
  │   ◄─ message_end        │                         │
  │                         │   streaming chunks      │
  │   ◄─ message_start      │ ◄─────────────────────  │
  │   ◄─ message_update     │   "I'll read that..."   │
  │   ◄─ message_update     │   tool_call: read_file  │
  │   ◄─ message_end        │                         │
  │                         │                         │
  │   ◄─ tool_execution_start │                        │
  │   ◄─ tool_execution_end   │  (executes read_file)  │
  │                         │                         │
  │   ◄─ turn_end           │                         │
  │                         │  messages + tool result  │
  │   ◄─ turn_start         │ ───────────────────────►│
  │                         │                         │
  │   ◄─ message_start      │   streaming chunks      │
  │   ◄─ message_update     │ ◄─────────────────────  │
  │   ◄─ message_update     │   "Here's what I found" │
  │   ◄─ message_end        │   (no more tool calls)  │
  │                         │                         │
  │   ◄─ turn_end           │                         │
  │   ◄─ agent_end          │                         │
  │                         │                         │
```

The consumer (your app) gets a stream of events. What it does with them is its
problem:

```python
# CLI — print tokens as they arrive
async for event in agent.prompt("read main.py"):
    if event["type"] == "message_update" and event["delta_type"] == "text_delta":
        print(event["delta"].content, end="", flush=True)

# FastHTML — SSE to browser
async for event in agent.prompt("read main.py"):
    yield sse(render_event(event))

# Slack — post final message
result = await agent.prompt("read main.py").result()
slack.post(extract_text(result))
```

Same agent, same loop, same events. Different consumers.

## What We're NOT Building

This matters as much as what we are building:

| Not included | Why | Who handles it |
|---|---|---|
| Tools | Consumer defines them | Your app |
| System prompt | Consumer writes it | Your app |
| UI | Not our problem | Your app |
| Persistence | Not our problem | Your app |
| Context compaction | Hook exists (`transform_context`) | Your app |
| MCP support | Not needed yet | Maybe later |
| Sub-agents | Tools can spawn agents | Naturally composable |
| Automatic retry | Consumer decides when | `continue_run()` |


## The Tool Protocol

Tools are simple. Define a name, description, JSON schema, and an async
function:

```python
Tool(
    name="read_file",
    description="Read a file from disk",
    parameters={
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
    },
    execute=read_file_fn,
)
```

The execute function gets four arguments:

```python
async def read_file_fn(tool_call_id, params, signal, on_update):
    content = Path(params["path"]).read_text()
    return ToolResult(content=[{"type": "text", "text": content}])
```

- **`tool_call_id`** — unique ID from the LLM, for event routing
- **`params`** — validated arguments (Pydantic coerces types: `"42"` → `42`)
- **`signal`** — `asyncio.Event` for cancellation (check `signal.is_set()`)
- **`on_update`** — callback for streaming partial results

Tools can return text, images, or both. The loop handles the rest.

```python
# Text + image (e.g., matplotlib chart)
return ToolResult(
    content=[
        {"type": "text", "text": "Generated chart"},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
    ]
)
```

## The EventStream

Everything flows through one primitive: an async queue.

```python
stream = agent.prompt("do something")

async for event in stream:      # consumer iterates
    handle(event)               # events arrive as they happen

result = await stream.result()  # or just wait for the final answer
```

The loop pushes events. The consumer pulls them. That's the interface. It works
with `async for` (real-time streaming) or `await result()` (batch). The
EventStream decouples production from consumption — the loop doesn't know or
care what's consuming the events.

Under the hood it's an `asyncio.Queue` with a sentinel value for completion:

```
Producer (loop)                Consumer (your app)
     │                              │
     │── push(event) ──►  Queue  ──►│── async for event in stream
     │── push(event) ──►         ──►│
     │── push(event) ──►         ──►│
     │── end(result)  ──► None   ──►│── (iteration ends)
     │                              │
```


## The Build Plan

We work from the inside out, one piece at a time:

```
 1. EventStream ──► 2. Types ──► 3. Loop ──► 4. Agent ──► 5. Tests
    (the queue)     (vocabulary)  (engine)    (wrapper)    (proof)
```

At each step: write it, test it, understand it. 