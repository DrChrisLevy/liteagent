# liteagent

Minimal async agent loop with streaming, steering, and tool execution.

Inspired by [pi-mono](https://github.com/badlogic/pi-mono)'s agent loop — same architecture, Python idioms. Uses [litellm](https://github.com/BerriAI/litellm) for LLM access, so it works with any provider (Anthropic, OpenAI, Google, etc.).

## Install

```bash
pip install git+https://github.com/DrChrisLevy/liteagent.git
```

## Quick start

```python
import asyncio
from liteagent import Agent, Tool, ToolResult


# Define a tool
async def greet_execute(tool_call_id, params, signal, on_update):
    name = params["name"]
    return ToolResult(content=[{"type": "text", "text": f"Hello, {name}!"}])


greet = Tool(
    name="greet",
    description="Greet someone by name",
    parameters={
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
    },
    execute=greet_execute,
)

# Create an agent
agent = Agent(
    model="anthropic/claude-sonnet-4-6",
    tools=[greet],
    system_prompt="You are a helpful assistant. Use the greet tool when asked.",
)


# Subscribe to events
def on_event(event):
    print(event)


agent.subscribe(on_event)


# Run it
async def main():
    await agent.prompt("Say hi to Alice")


asyncio.run(main())
```

## What it does

- **Dual loop architecture** — inner loop for LLM + tool calls, outer loop for follow-ups
- **Streaming** — token-by-token events via async iteration
- **Steering** — interrupt the agent mid-run with new instructions
- **Follow-ups** — queue messages while the agent is busy
- **Sequential tool execution** — tools run one at a time so steering can interrupt between them
- **Any LLM** — works with any model litellm supports

## Core API

### `Agent`

```python
agent = Agent(
    model="anthropic/claude-sonnet-4-6",  # any litellm model string
    tools=[...],                           # list of Tool objects
    system_prompt="...",                   # system prompt
    convert_to_llm=None,                  # custom message converter (optional)
    transform_context=None,               # context transform hook (optional)
)

await agent.prompt("hello")               # send a message, run until complete
await agent.continue_run()                # resume with existing tool results
agent.steer("do this instead")            # interrupt mid-run
agent.follow_up("also do this")           # queue for after current run
agent.abort()                             # cancel the current run
await agent.wait_for_idle()               # wait until agent finishes
agent.subscribe(callback)                 # subscribe to events
```

### `Tool`

```python
from pydantic import BaseModel

class SearchParams(BaseModel):
    query: str
    max_results: int = 5

search = Tool(
    name="search",
    description="Search the web",
    parameters=SearchParams.model_json_schema(),
    params_model=SearchParams,  # optional: validates + coerces arguments
    execute=my_search_function,  # async def(id, params, signal, on_update) -> ToolResult
)
```

### Raw loop functions

For direct control without the `Agent` wrapper:

```python
from liteagent import agent_loop, agent_loop_continue, make_default_convert, EventStream, AgentConfig, AgentContext

stream = EventStream()
config = AgentConfig(model="anthropic/claude-sonnet-4-6", convert_to_llm=make_default_convert("anthropic/claude-sonnet-4-6"))
context = AgentContext(system_prompt="...", messages=[...], tools=[...])

await agent_loop(stream, config, context, signal=signal)
```

### Events

The event stream emits these event types:

| Event | When |
|---|---|
| `agent_start` / `agent_end` | Agent run begins / ends |
| `turn_start` / `turn_end` | Each LLM call begins / ends |
| `message_start` / `message_update` / `message_end` | Message lifecycle |
| `tool_execution_start` / `tool_execution_update` / `tool_execution_end` | Tool execution lifecycle |

`message_update` events include a `delta_type` field: `text_delta`, `thinking_delta`, or `tool_call_delta`.

## Auth

Set API keys as environment variables — litellm reads them automatically:

```bash
export ANTHROPIC_API_KEY=sk-...
export OPENAI_API_KEY=sk-...
export GEMINI_API_KEY=...
```

## Design


- `stream.py` — async event stream (producer-consumer queue)
- `types.py` — shared types (Tool, ToolResult, AgentConfig, etc.)
- `loop.py` — stateless dual loop (LLM calls, tool execution, steering)
- `convert.py` — default message converter (sole provider-specific boundary)
- `agent.py` — stateful wrapper (message history, queues, subscriptions)

See [DESIGN_NOTES.md](DESIGN_NOTES.md) for architecture decisions and pi-mono comparison.

## License

MIT
