# liteagent

Minimal async agent loop with streaming, steering, and tool execution. Ported from [pi-mono](https://github.com/badlogic/pi-mono)'s TypeScript agent core — same dual-loop architecture, Python idioms. Uses [litellm](https://github.com/BerriAI/litellm) for LLM access, so it works with any provider.

This library gives you the loop primitives — LLM calls, tool execution, event streaming, steering, follow-ups. It does not own your tools, your UI, your transport layer, or your persistence. You build the product/consumers/applications on top.

## Setup

```bash
uv add "liteagent @ git+https://github.com/DrChrisLevy/liteagent.git"
```

Set at least one API key. For example, create a `.env` file:

```
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
GEMINI_API_KEY=...
```

Then load it before using liteagent.  For example,

```python
from dotenv import load_dotenv
load_dotenv()
```

## Basic prompt

```python
import asyncio
from liteagent import Agent

agent = Agent(
    model="anthropic/claude-sonnet-4-6",
    system_prompt="Be concise.",
)

asyncio.run(agent.prompt("What is 2 + 2?"))

print(agent.messages)
print(agent.state)
```

Any [litellm model string](https://docs.litellm.ai/docs/providers) works: `gpt-5.2`, `gemini/gemini-3-flash-preview`, etc.

## Tools

Tools must be **async** functions that return a `ToolResult`. The LLM decides when to call them.

```python
from liteagent import Agent, Tool, ToolResult

async def get_weather(tool_call_id, params, signal, on_update):
    city = params["city"]
    # In reality, call a weather API here
    return ToolResult(content=[{"type": "text", "text": f"{city}: 72°F, sunny"}])

weather = Tool(
    name="get_weather",
    description="Get current weather for a city",
    parameters={
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    },
    execute=get_weather,
)

agent = Agent(
    model="anthropic/claude-sonnet-4-6",
    tools=[weather],
    system_prompt="Use the weather tool when asked about weather.",
)

asyncio.run(agent.prompt("What's the weather in Paris?"))
print(agent.messages)
print(agent.state)
```

The agent calls the tool, gets the result, and responds to the user — all in one `prompt()` call. Tools run sequentially so the agent can be interrupted between them (see [Steering](#steering)).

## Pydantic validation

Add a `params_model` to validate and coerce tool arguments before execution:

```python
from pydantic import BaseModel

class WeatherParams(BaseModel):
    city: str
    units: str = "fahrenheit"

weather = Tool(
    name="get_weather",
    description="Get current weather for a city",
    parameters=WeatherParams.model_json_schema(),
    params_model=WeatherParams,  # validates + coerces args
    execute=get_weather,
)
```

LLMs sometimes send `"42"` when the schema says `int` — Pydantic coerces it automatically. If validation fails entirely, the error becomes a tool result with `is_error=True`, giving the LLM a chance to retry with corrected arguments.

## Events and subscribers

Subscribers are the primary way to consume the agent. Your callback fires in real-time as events arrive — during streaming, not after.

```python
agent = Agent(model="anthropic/claude-sonnet-4-6", tools=[weather])

# Events fire live during prompt() — this prints as tokens stream in
def on_event(event):
    # stream/consume the event
    print(event)

agent.subscribe(on_event)
asyncio.run(agent.prompt("What's the weather in Paris?"))
```

A typical tool-calling run produces this event sequence:

```
agent_start
turn_start
message_start                      # user message
message_end                        # user message
message_start                      # assistant starts streaming
message_update     text_delta      # "Let me check..."
message_update     text_delta      # " the weather in Paris"
message_update     tool_call_delta # tool call id + name
message_update     tool_call_delta # tool call arguments streaming
message_end                        # assistant finalized (has text + tool_calls)
tool_execution_start               # get_weather begins
tool_execution_end                 # get_weather returns result
message_start                      # tool result message
message_end                        # tool result message
turn_end
turn_start                         # second LLM call with tool result
message_start                      # assistant starts streaming
message_update     text_delta      # "The current weather..."
message_update     text_delta      # "in Paris is 72°F"
message_end                        # assistant finalized (stop_reason="stop")
turn_end
agent_end
```

Here's a runnable CLI consumer with two tools:

```python
import asyncio
from liteagent import Agent, Tool, ToolResult

async def get_weather(tool_call_id, params, signal, on_update):
    city = params["city"]
    return ToolResult(content=[{"type": "text", "text": f"{city}: 72°F, sunny"}])

async def get_time(tool_call_id, params, signal, on_update):
    city = params["city"]
    return ToolResult(content=[{"type": "text", "text": f"{city}: 2:30 PM"}])

weather = Tool(
    name="get_weather", description="Get current weather",
    parameters={"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
    execute=get_weather,
)
time_tool = Tool(
    name="get_time", description="Get current time",
    parameters={"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
    execute=get_time,
)

agent = Agent(
    model="anthropic/claude-sonnet-4-6",
    tools=[weather, time_tool],
    system_prompt="Use tools when asked. Be concise.",
)

def on_event(event):
    t = event["type"]
    if t == "message_update" and event["delta_type"] == "text_delta":
        print(event["delta"]["content"], end="", flush=True)
    elif t == "tool_execution_start":
        print(f"\n[calling {event['tool_name']}({event['args']})]")
    elif t == "tool_execution_end":
        print(f"[result: {event['result']['content'][0]['text']}]")
    elif t == "agent_end":
        print()

unsub = agent.subscribe(on_event) # can unsubscribe later
asyncio.run(agent.prompt("What's the weather and time in Paris?"))
```


Unsubscribe when done:

```python
unsub()  # stop receiving events
asyncio.run(agent.prompt("What's the weather and time in Toronto"))
```

## Agent state

The agent tracks its state as it runs. This is what you inspect between or during calls:

```python
agent.state.is_streaming      # True while prompt() is running
agent.state.model             # current model string
agent.state.system_prompt     # current system prompt
agent.state.tools             # registered tools
agent.state.thinking_level    # "off", "low", "medium", "high"
agent.state.stream_message    # partial message during streaming (None when idle)
agent.state.pending_tool_calls  # set of tool call IDs currently executing
agent.state.error             # last error message, or None
```

`agent.messages` is the full conversation history — a list of plain dicts. Mutate when needed:

```python
agent.messages                    # read the list
agent.append_message(msg)         # add one message
agent.replace_messages(new_list)  # swap entire history
agent.clear_messages()            # empty history (keeps queues/error)
agent.reset()                     # clear everything, keep config
```

## Steering

Interrupt the agent mid-run with new instructions. Tools run one at a time, and between each tool the agent checks for steering messages:

```python
import asyncio
from liteagent import Agent, Tool, ToolResult

async def slow_search(tool_call_id, params, signal, on_update):
    await asyncio.sleep(2)  # simulate slow work
    return ToolResult(content=[{"type": "text", "text": f"Results for: {params['q']}"}])

search = Tool(
    name="search", description="Search", execute=slow_search,
    parameters={"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]},
)

agent = Agent(model="anthropic/claude-sonnet-4-6", tools=[search])

# Steer once after the first tool finishes — remaining tools get skipped
state = {"steered": False}
def on_event(event):
    if event["type"] == "tool_execution_end" and not state["steered"]:
        state["steered"] = True
        agent.steer("Actually, forget that. Just say hello.")


agent.subscribe(on_event)
asyncio.run(agent.prompt("Search for 'python' and 'rust' and 'go'"))
print(agent.messages)
```

The agent executes the first search, sees the steering message, skips the remaining searches, and follows the new instruction instead.

## Follow-ups

Queue messages that get delivered after the current run finishes:

```python
agent = Agent(model="anthropic/claude-sonnet-4-6")
agent.follow_up("Now translate that to French")
asyncio.run(agent.prompt("Write a one-sentence summary of Python"))

# Agent answers the prompt, then automatically handles the follow-up
print(agent.messages)
```

## Multimodal

Send images in prompts:

```python
agent = Agent(model="anthropic/claude-sonnet-4-6")
asyncio.run(agent.prompt(
    "What's in this image?",
    images=[{
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64,iVBOR..."},
    }],
))
```

Return images from tools — works across all providers (OpenAI image handling is automatic):

```python
async def generate_chart(tool_call_id, params, signal, on_update):
    img_b64 = create_chart()  # your chart logic
    return ToolResult(content=[
        {"type": "text", "text": "Here's the chart."},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
    ])
```

## Thinking / reasoning

Enable extended thinking with one line:

```python
agent = Agent(model="anthropic/claude-sonnet-4-6")
agent.set_thinking_level("high")  # "low", "medium", "high"
asyncio.run(agent.prompt("What is 17 * 23? Think step by step."))
print(agent.messages)
```

Thinking tokens stream as `thinking_delta` events. Works with Anthropic and Gemini. Thinking metadata (signatures, blocks) is preserved automatically for multi-turn conversations.

## Context transform

Inject, prune, or compact messages before each LLM call:

```python
def my_transform(messages, signal=None):
    # Example: keep only the last 10 messages
    return messages[-10:]

agent = Agent(
    model="anthropic/claude-sonnet-4-6",
    transform_context=my_transform,
)
```

## Abort

Cancel a running agent:

```python
def on_event(event):
    if event["type"] == "message_update":
        agent.abort()  # cancel immediately

agent.subscribe(on_event)
asyncio.run(agent.prompt("Write a long essay"))
# Agent stops, partial content is preserved if meaningful
```

## Multi-turn

Messages accumulate automatically across `prompt()` calls:

```python
agent = Agent(model="anthropic/claude-sonnet-4-6")
asyncio.run(agent.prompt("My name is Alice"))
asyncio.run(agent.prompt("What's my name?"))
# Agent remembers: "Alice"

agent.reset()  # clear history, keep config
```

## License

MIT
