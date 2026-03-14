---
title: Part 2 of Porting pi-mono's agent loop to Python -  The Dual While Loop Engine
author: Chris Levy
draft: false
date: '2026-03-11'
date-modified: '2026-03-11'
image: dual_loop.jpg
description: "Part 2 of porting pi-mono's agent loop to Python. A hands-on walkthrough of the core loop — LLM streaming, tool execution, steering, follow-ups, and everything in between."
tags:
  - python
  - async
  - agents
  - llm
---
# Intro

This is part 2 of a series where I port [pi-mono](https://github.com/badlogic/pi-mono)'s agent loop from TypeScript to Python as a learning exercise. In [Part 1](https://drchrislevy.com/blog/blog_post?fpath=posts%2Fasync_event_streams%2Fasync_event_streams.ipynb) we walked through Async and Event Streams i.e. `EventStream`. This was the async pipe that connects producers to consumers. 

When you call `agent.prompt("do something")`, a loop takes over. It calls the LLM, streams the response token by token, executes tools, checks for user interruptions, and repeats until the LLM
is done. 

In this post we will work with the loop layer directly. We will interact with `agent_loop()` and `agent_loop_continue()` instead of going through the Agent class. You wouldn't do this in practice. The `Agent` is the public facing API. The `Agent` class (Part 3 blog post to come) adds state management and queues on top. But the interesting mechanics all live here. This is an educational post to see what is happening under the hood.

We will walkthrough the functionality of `liteagent.loop`.
We'll call the loop directly, watch events stream, see tool execution, and understand each moving part.
Here are some topics we'll cover:

1. Why does `convert_to_llm` exist? (internal format vs what litellm accepts)
2. Simplest possible call — one LLM round, no tools
3. Adding a tool — echo
4. Error handling — tools that throw
5. Pydantic validation + type coercion
6. Multi-turn context — how messages accumulate
7. The `on_update` callback — streaming from tools
8. Multiple tool calls
9. Usage tracking
10. `agent_loop_continue` — resuming from manually-built context
11. Testing with different models
12. Steering — interrupting the loop mid-run
13. Follow-ups — the outer loop
14. Cancellation — `signal`
15. `transform_context` — modifying messages before each LLM call
16. `reasoning_effort` — thinking/reasoning
17. `ToolResult.details` — UI-only data
18. Multimodal tools + `make_default_convert`


## Setup

The loop has **two entry points** (like pi's `agentLoop` and `agentLoopContinue`):

- `agent_loop(prompts, context, config, signal)` — start a new run with prompt messages
- `agent_loop_continue(context, config, signal)` — resume from existing context

Both return an `EventStream` immediately. The actual LLM call runs as an async task.

To use them, we need:

- **`AgentContext`** — system prompt + messages + tools (the *data* the loop works on)
- **`AgentConfig`** — model + convert_to_llm + hooks (the *behavior* of the loop)


```python
from liteagent import (
    agent_loop,
    agent_loop_continue,
    make_default_convert,
    AgentContext,
    AgentConfig,
    Tool,
    ToolResult,
)

ALL_MODELS = [
    "anthropic/claude-sonnet-4-6",
    "anthropic/claude-opus-4-6",
    "gemini/gemini-3-flash-preview",
    "gpt-5.2",
]
```

## 1. Why does `convert_to_llm` exist?

I chose to use [litellm](https://docs.litellm.ai/docs/) as the LLM library.
But regardless of the library used, the loop needs to *translate* our messages into the provider's format.
[litellm](https://docs.litellm.ai/docs/) accepts OpenAI-format messages and translates them for each provider. So why do we need a `convert_to_llm` hook at all? Can't we just send our messages straight through?

Let's find out by looking at what the loop *actually stores* on messages.
As of writing, the messages are plain dicts, not typed dicts or dataclasses.
I might change this in the future.


```python
import json

async def echo_fn(tool_call_id, params, signal=None, on_update=None):
    return ToolResult(content=[{"type": "text", "text": params["message"]}])

echo_tool = Tool(
    name="echo", description="Echo a message",
    parameters={"type": "object", "properties": {"message": {"type": "string"}}, "required": ["message"]},
    execute=echo_fn,
)

MODEL = "anthropic/claude-sonnet-4-6"
context = AgentContext(system_prompt="Use the echo tool.", messages=[], tools=[echo_tool])
config = AgentConfig(model=MODEL, convert_to_llm=make_default_convert(MODEL))

stream = agent_loop([{"role": "user", "content": "Echo 'hello'"}], context, config)
async for event in stream:
    if event["type"] == "agent_end":
        for msg in event["messages"]:
            print(f"\n=== {msg['role']} ===")
            print(json.dumps(msg, indent=2, default=str))
```

    
    === user ===
    {
      "role": "user",
      "content": "Echo 'hello'"
    }
    
    === assistant ===
    {
      "role": "assistant",
      "content": null,
      "tool_calls": [
        {
          "id": "toolu_01SHP315kkJwEvqVKN1Bodfg",
          "type": "function",
          "function": {
            "name": "echo",
            "arguments": "{\"message\": \"hello\"}"
          },
          "provider_specific_fields": null
        }
      ],
      "thinking_blocks": null,
      "reasoning_content": null,
      "provider_specific_fields": null,
      "model": "anthropic/claude-sonnet-4-6",
      "usage": {
        "prompt_tokens": 564,
        "completion_tokens": 52,
        "total_tokens": 616,
        "cache_read_tokens": 0,
        "cache_creation_tokens": 0
      },
      "stop_reason": "tool_calls",
      "timestamp": 1773365234805
    }
    
    === tool ===
    {
      "role": "tool",
      "tool_call_id": "toolu_01SHP315kkJwEvqVKN1Bodfg",
      "name": "echo",
      "content": [
        {
          "type": "text",
          "text": "hello"
        }
      ],
      "details": {},
      "is_error": false,
      "timestamp": 1773365234805
    }
    
    === assistant ===
    {
      "role": "assistant",
      "content": "The tool echoed back: **hello**! Let me know if there's anything else you'd like to do.",
      "tool_calls": null,
      "thinking_blocks": null,
      "reasoning_content": null,
      "provider_specific_fields": null,
      "model": "anthropic/claude-sonnet-4-6",
      "usage": {
        "prompt_tokens": 629,
        "completion_tokens": 27,
        "total_tokens": 656,
        "cache_read_tokens": 0,
        "cache_creation_tokens": 0
      },
      "stop_reason": "stop",
      "timestamp": 1773365236475
    }


### So `convert_to_llm` exists for two reasons:

**Reason 1: Strip our extras.** The loop enriches messages with metadata the LLM can't see
(`usage`, `stop_reason`, `timestamp`, `details`, `is_error`, etc.). These need to be stripped
before sending to the provider.

**Reason 2: Provider quirks.** For example, OpenAI requires tool results as plain strings, not content
block arrays. When a tool returns images, OpenAI silently drops them from tool results —
so they must be hoisted into synthetic user messages. Anthropic/Gemini accept content blocks
with images natively. `convert_to_llm` handles this so the loop stays provider-agnostic.

### The built-in default: `make_default_convert`

liteagent ships a default converter (`liteagent.convert.make_default_convert`) that handles
both of these. It uses a **denylist** approach — strips the known liteagent metadata fields
and passes everything else through (so new litellm fields like `provider_specific_fields`
work automatically). It also handles OpenAI image hoisting.

```python
from liteagent import make_default_convert

convert = make_default_convert("anthropic/claude-sonnet-4-6")  # returns a function
```

The `Agent` class uses this by default. You only need to pass `convert_to_llm` if you want
custom behavior (e.g., mapping app-specific message types like `bashExecution` → user messages).

Pi's coding agent does exactly this: overrides the converter to map custom message types
(`bashExecution`, `branchSummary`, etc.) into standard `user`/`assistant`/`tool` messages.

We will dive more into this in Part 3 blog post.


```python
event["messages"]
```




    [{'role': 'user', 'content': "Echo 'hello'"},
     {'role': 'assistant',
      'content': None,
      'tool_calls': [{'id': 'toolu_01SHP315kkJwEvqVKN1Bodfg',
        'type': 'function',
        'function': {'name': 'echo', 'arguments': '{"message": "hello"}'},
        'provider_specific_fields': None}],
      'thinking_blocks': None,
      'reasoning_content': None,
      'provider_specific_fields': None,
      'model': 'anthropic/claude-sonnet-4-6',
      'usage': {'prompt_tokens': 564,
       'completion_tokens': 52,
       'total_tokens': 616,
       'cache_read_tokens': 0,
       'cache_creation_tokens': 0},
      'stop_reason': 'tool_calls',
      'timestamp': 1773365234805},
     {'role': 'tool',
      'tool_call_id': 'toolu_01SHP315kkJwEvqVKN1Bodfg',
      'name': 'echo',
      'content': [{'type': 'text', 'text': 'hello'}],
      'details': {},
      'is_error': False,
      'timestamp': 1773365234805},
     {'role': 'assistant',
      'content': "The tool echoed back: **hello**! Let me know if there's anything else you'd like to do.",
      'tool_calls': None,
      'thinking_blocks': None,
      'reasoning_content': None,
      'provider_specific_fields': None,
      'model': 'anthropic/claude-sonnet-4-6',
      'usage': {'prompt_tokens': 629,
       'completion_tokens': 27,
       'total_tokens': 656,
       'cache_read_tokens': 0,
       'cache_creation_tokens': 0},
      'stop_reason': 'stop',
      'timestamp': 1773365236475}]




```python
# Demo: what the default converter does to our enriched messages
convert = make_default_convert("anthropic/claude-sonnet-4-6")
convert(event["messages"])
```




    [{'role': 'user', 'content': "Echo 'hello'"},
     {'role': 'assistant',
      'content': None,
      'tool_calls': [{'id': 'toolu_01Kh4z8YmX3kMQNZuxtGGQfS',
        'type': 'function',
        'function': {'name': 'echo', 'arguments': '{"message": "hello"}'},
        'provider_specific_fields': None}],
      'thinking_blocks': None,
      'reasoning_content': None,
      'provider_specific_fields': None,
      'model': 'anthropic/claude-sonnet-4-6'},
     {'role': 'tool',
      'tool_call_id': 'toolu_01Kh4z8YmX3kMQNZuxtGGQfS',
      'name': 'echo',
      'content': [{'type': 'text', 'text': 'hello'}]},
     {'role': 'assistant',
      'content': "The tool echoed back: **hello**! Let me know if there's anything else you'd like to do.",
      'tool_calls': None,
      'thinking_blocks': None,
      'reasoning_content': None,
      'provider_specific_fields': None,
      'model': 'anthropic/claude-sonnet-4-6'}]



## 2. Simplest possible call — one LLM round, no tools

Let's use `make_default_convert` and make a simple call. The absolute minimum:
send a prompt, get a response, watch the events stream back.


```python
# MODEL — change this to test different providers
MODEL = "anthropic/claude-sonnet-4-6"

context = AgentContext(
    system_prompt="You are a helpful assistant. Be concise.",
    messages=[],
    tools=None,
)

config = AgentConfig(
    model=MODEL,
    convert_to_llm=make_default_convert(MODEL),
)

# agent_loop takes a list of messages to inject — any role works,
# but in practice these are user messages (the new prompt).
prompt_messages = [{"role": "user", "content": "What is 2+2? Answer in several words."}]

stream = agent_loop(prompt_messages, context, config)

# Consume the stream — every event printed
events = []
async for event in stream:
    print(event)
```

    {'type': 'agent_start'}
    {'type': 'turn_start'}
    {'type': 'message_start', 'message': {'role': 'user', 'content': 'What is 2+2? Answer in several words.'}}
    {'type': 'message_end', 'message': {'role': 'user', 'content': 'What is 2+2? Answer in several words.'}}
    {'type': 'message_start', 'message': {'role': 'assistant', 'content': None, 'tool_calls': None}}
    {'type': 'message_update', 'message': {'role': 'assistant', 'content': 'The', 'tool_calls': None}, 'delta': {'content': 'The'}, 'delta_type': 'text_delta'}
    {'type': 'message_update', 'message': {'role': 'assistant', 'content': 'The answer is simply', 'tool_calls': None}, 'delta': {'content': ' answer is simply'}, 'delta_type': 'text_delta'}
    {'type': 'message_update', 'message': {'role': 'assistant', 'content': 'The answer is simply **4**.', 'tool_calls': None}, 'delta': {'content': ' **4**.'}, 'delta_type': 'text_delta'}
    {'type': 'message_end', 'message': {'role': 'assistant', 'content': 'The answer is simply **4**.', 'tool_calls': None, 'thinking_blocks': None, 'reasoning_content': None, 'provider_specific_fields': None, 'model': 'anthropic/claude-sonnet-4-6', 'usage': {'prompt_tokens': 30, 'completion_tokens': 10, 'total_tokens': 40, 'cache_read_tokens': 0, 'cache_creation_tokens': 0}, 'stop_reason': 'stop', 'timestamp': 1773365264158}}
    {'type': 'turn_end', 'message': {'role': 'assistant', 'content': 'The answer is simply **4**.', 'tool_calls': None, 'thinking_blocks': None, 'reasoning_content': None, 'provider_specific_fields': None, 'model': 'anthropic/claude-sonnet-4-6', 'usage': {'prompt_tokens': 30, 'completion_tokens': 10, 'total_tokens': 40, 'cache_read_tokens': 0, 'cache_creation_tokens': 0}, 'stop_reason': 'stop', 'timestamp': 1773365264158}, 'tool_results': []}
    {'type': 'agent_end', 'messages': [{'role': 'user', 'content': 'What is 2+2? Answer in several words.'}, {'role': 'assistant', 'content': 'The answer is simply **4**.', 'tool_calls': None, 'thinking_blocks': None, 'reasoning_content': None, 'provider_specific_fields': None, 'model': 'anthropic/claude-sonnet-4-6', 'usage': {'prompt_tokens': 30, 'completion_tokens': 10, 'total_tokens': 40, 'cache_read_tokens': 0, 'cache_creation_tokens': 0}, 'stop_reason': 'stop', 'timestamp': 1773365264158}]}


### What just happened — event sequence

For a simple no-tools call, the event sequence is:

```markdown
agent_start          — loop begins
turn_start           — first LLM call
message_start        — user prompt echoed
message_end          — user prompt done
message_start        — assistant starts streaming
message_update (x N) — text deltas arrive token by token
message_end          — assistant message finalized
turn_end             — turn complete (no tool results)
agent_end            — loop done, messages returned
```

The stream's `.result()` gives you all new messages from this run.


```python
# The final result — all **new** messages from this run
result = await stream.result()
result
```




    [{'role': 'user', 'content': 'What is 2+2? Answer in several words.'},
     {'role': 'assistant',
      'content': 'The answer is simply **4**.',
      'tool_calls': None,
      'thinking_blocks': None,
      'reasoning_content': None,
      'provider_specific_fields': None,
      'model': 'anthropic/claude-sonnet-4-6',
      'usage': {'prompt_tokens': 30,
       'completion_tokens': 10,
       'total_tokens': 40,
       'cache_read_tokens': 0,
       'cache_creation_tokens': 0},
      'stop_reason': 'stop',
      'timestamp': 1773365264158}]



## 3. Adding a tool — echo

Tools are `Tool` dataclasses with:
- `name`, `description`, `parameters` — what the LLM sees (JSON Schema)
- `execute` — what we call: `async def(tool_call_id, params, signal, on_update) -> ToolResult`
- `params_model` (optional) — Pydantic BaseModel for validation + type coercion

When the LLM decides to call a tool, the loop:
1. Parses the JSON arguments
2. Validates with Pydantic (if `params_model` set)
3. Calls `execute()`
4. Wraps the result as a tool message
5. Sends it back to the LLM for the next turn


```python
# Define the echo tool
async def echo_execute(tool_call_id, params, signal=None, on_update=None):
    return ToolResult(content=[{"type": "text", "text": params["message"]}])


echo_tool = Tool(
    name="echo",
    description="Echo back the given message. Use this when asked to echo something.",
    parameters={
        "type": "object",
        "properties": {
            "message": {"type": "string", "description": "The message to echo back"}
        },
        "required": ["message"],
    },
    execute=echo_execute,
)

print(f"Tool defined: {echo_tool.name}")
print(f"Parameters schema: {json.dumps(echo_tool.parameters, indent=2)}")
```

    Tool defined: echo
    Parameters schema: {
      "type": "object",
      "properties": {
        "message": {
          "type": "string",
          "description": "The message to echo back"
        }
      },
      "required": [
        "message"
      ]
    }



```python
context = AgentContext(
    system_prompt="You are helpful. When asked to echo, use the echo tool.",
    messages=[],
    tools=[echo_tool],
)
config = AgentConfig(model=MODEL, convert_to_llm=make_default_convert(MODEL))

prompt = [{"role": "user", "content": "Echo the message: hello world"}]
stream = agent_loop(prompt, context, config)

events = []
async for event in stream:
    print(event)
```

    {'type': 'agent_start'}
    {'type': 'turn_start'}
    {'type': 'message_start', 'message': {'role': 'user', 'content': 'Echo the message: hello world'}}
    {'type': 'message_end', 'message': {'role': 'user', 'content': 'Echo the message: hello world'}}
    {'type': 'message_start', 'message': {'role': 'assistant', 'content': None, 'tool_calls': None}}
    {'type': 'message_update', 'message': {'role': 'assistant', 'content': None, 'tool_calls': [{'id': 'toolu_01XSaB8TT1b999xzcepgxS8J', 'type': 'function', 'function': {'name': 'echo', 'arguments': ''}}]}, 'delta': {'tool_calls': [{'index': 0, 'id': 'toolu_01XSaB8TT1b999xzcepgxS8J', 'function': {'name': 'echo'}}]}, 'delta_type': 'tool_call_delta'}
    {'type': 'message_update', 'message': {'role': 'assistant', 'content': None, 'tool_calls': [{'id': 'toolu_01XSaB8TT1b999xzcepgxS8J', 'type': 'function', 'function': {'name': 'echo', 'arguments': ''}}]}, 'delta': {'tool_calls': [{'index': 0}]}, 'delta_type': 'tool_call_delta'}
    {'type': 'message_update', 'message': {'role': 'assistant', 'content': None, 'tool_calls': [{'id': 'toolu_01XSaB8TT1b999xzcepgxS8J', 'type': 'function', 'function': {'name': 'echo', 'arguments': '{"me'}}]}, 'delta': {'tool_calls': [{'index': 0, 'function': {'arguments': '{"me'}}]}, 'delta_type': 'tool_call_delta'}
    {'type': 'message_update', 'message': {'role': 'assistant', 'content': None, 'tool_calls': [{'id': 'toolu_01XSaB8TT1b999xzcepgxS8J', 'type': 'function', 'function': {'name': 'echo', 'arguments': '{"message": "hel'}}]}, 'delta': {'tool_calls': [{'index': 0, 'function': {'arguments': 'ssage": "hel'}}]}, 'delta_type': 'tool_call_delta'}
    {'type': 'message_update', 'message': {'role': 'assistant', 'content': None, 'tool_calls': [{'id': 'toolu_01XSaB8TT1b999xzcepgxS8J', 'type': 'function', 'function': {'name': 'echo', 'arguments': '{"message": "hello wo'}}]}, 'delta': {'tool_calls': [{'index': 0, 'function': {'arguments': 'lo wo'}}]}, 'delta_type': 'tool_call_delta'}
    {'type': 'message_update', 'message': {'role': 'assistant', 'content': None, 'tool_calls': [{'id': 'toolu_01XSaB8TT1b999xzcepgxS8J', 'type': 'function', 'function': {'name': 'echo', 'arguments': '{"message": "hello world"}'}}]}, 'delta': {'tool_calls': [{'index': 0, 'function': {'arguments': 'rld"}'}}]}, 'delta_type': 'tool_call_delta'}
    {'type': 'message_end', 'message': {'role': 'assistant', 'content': None, 'tool_calls': [{'id': 'toolu_01XSaB8TT1b999xzcepgxS8J', 'type': 'function', 'function': {'name': 'echo', 'arguments': '{"message": "hello world"}'}, 'provider_specific_fields': None}], 'thinking_blocks': None, 'reasoning_content': None, 'provider_specific_fields': None, 'model': 'anthropic/claude-sonnet-4-6', 'usage': {'prompt_tokens': 594, 'completion_tokens': 53, 'total_tokens': 647, 'cache_read_tokens': 0, 'cache_creation_tokens': 0}, 'stop_reason': 'tool_calls', 'timestamp': 1773313747681}}
    {'type': 'tool_execution_start', 'tool_call_id': 'toolu_01XSaB8TT1b999xzcepgxS8J', 'tool_name': 'echo', 'args': {'message': 'hello world'}}
    {'type': 'tool_execution_end', 'tool_call_id': 'toolu_01XSaB8TT1b999xzcepgxS8J', 'tool_name': 'echo', 'result': {'content': [{'type': 'text', 'text': 'hello world'}], 'details': None}, 'is_error': False}
    {'type': 'message_start', 'message': {'role': 'tool', 'tool_call_id': 'toolu_01XSaB8TT1b999xzcepgxS8J', 'name': 'echo', 'content': [{'type': 'text', 'text': 'hello world'}], 'details': {}, 'is_error': False, 'timestamp': 1773313747681}}
    {'type': 'message_end', 'message': {'role': 'tool', 'tool_call_id': 'toolu_01XSaB8TT1b999xzcepgxS8J', 'name': 'echo', 'content': [{'type': 'text', 'text': 'hello world'}], 'details': {}, 'is_error': False, 'timestamp': 1773313747681}}
    {'type': 'turn_end', 'message': {'role': 'assistant', 'content': None, 'tool_calls': [{'id': 'toolu_01XSaB8TT1b999xzcepgxS8J', 'type': 'function', 'function': {'name': 'echo', 'arguments': '{"message": "hello world"}'}, 'provider_specific_fields': None}], 'thinking_blocks': None, 'reasoning_content': None, 'provider_specific_fields': None, 'model': 'anthropic/claude-sonnet-4-6', 'usage': {'prompt_tokens': 594, 'completion_tokens': 53, 'total_tokens': 647, 'cache_read_tokens': 0, 'cache_creation_tokens': 0}, 'stop_reason': 'tool_calls', 'timestamp': 1773313747681}, 'tool_results': [{'role': 'tool', 'tool_call_id': 'toolu_01XSaB8TT1b999xzcepgxS8J', 'name': 'echo', 'content': [{'type': 'text', 'text': 'hello world'}], 'details': {}, 'is_error': False, 'timestamp': 1773313747681}]}
    {'type': 'turn_start'}
    {'type': 'message_start', 'message': {'role': 'assistant', 'content': None, 'tool_calls': None}}
    {'type': 'message_update', 'message': {'role': 'assistant', 'content': 'The ech', 'tool_calls': None}, 'delta': {'content': 'The ech'}, 'delta_type': 'text_delta'}
    {'type': 'message_update', 'message': {'role': 'assistant', 'content': 'The echoed message is: **hello world**', 'tool_calls': None}, 'delta': {'content': 'oed message is: **hello world**'}, 'delta_type': 'text_delta'}
    {'type': 'message_end', 'message': {'role': 'assistant', 'content': 'The echoed message is: **hello world**', 'tool_calls': None, 'thinking_blocks': None, 'reasoning_content': None, 'provider_specific_fields': None, 'model': 'anthropic/claude-sonnet-4-6', 'usage': {'prompt_tokens': 661, 'completion_tokens': 13, 'total_tokens': 674, 'cache_read_tokens': 0, 'cache_creation_tokens': 0}, 'stop_reason': 'stop', 'timestamp': 1773313749377}}
    {'type': 'turn_end', 'message': {'role': 'assistant', 'content': 'The echoed message is: **hello world**', 'tool_calls': None, 'thinking_blocks': None, 'reasoning_content': None, 'provider_specific_fields': None, 'model': 'anthropic/claude-sonnet-4-6', 'usage': {'prompt_tokens': 661, 'completion_tokens': 13, 'total_tokens': 674, 'cache_read_tokens': 0, 'cache_creation_tokens': 0}, 'stop_reason': 'stop', 'timestamp': 1773313749377}, 'tool_results': []}
    {'type': 'agent_end', 'messages': [{'role': 'user', 'content': 'Echo the message: hello world'}, {'role': 'assistant', 'content': None, 'tool_calls': [{'id': 'toolu_01XSaB8TT1b999xzcepgxS8J', 'type': 'function', 'function': {'name': 'echo', 'arguments': '{"message": "hello world"}'}, 'provider_specific_fields': None}], 'thinking_blocks': None, 'reasoning_content': None, 'provider_specific_fields': None, 'model': 'anthropic/claude-sonnet-4-6', 'usage': {'prompt_tokens': 594, 'completion_tokens': 53, 'total_tokens': 647, 'cache_read_tokens': 0, 'cache_creation_tokens': 0}, 'stop_reason': 'tool_calls', 'timestamp': 1773313747681}, {'role': 'tool', 'tool_call_id': 'toolu_01XSaB8TT1b999xzcepgxS8J', 'name': 'echo', 'content': [{'type': 'text', 'text': 'hello world'}], 'details': {}, 'is_error': False, 'timestamp': 1773313747681}, {'role': 'assistant', 'content': 'The echoed message is: **hello world**', 'tool_calls': None, 'thinking_blocks': None, 'reasoning_content': None, 'provider_specific_fields': None, 'model': 'anthropic/claude-sonnet-4-6', 'usage': {'prompt_tokens': 661, 'completion_tokens': 13, 'total_tokens': 674, 'cache_read_tokens': 0, 'cache_creation_tokens': 0}, 'stop_reason': 'stop', 'timestamp': 1773313749377}]}


### Tool call event sequence

With a tool call, the loop does two turns:

```markdown
agent_start
turn_start                     ← turn 1: LLM decides to call a tool
  message_start (user)         ← prompt injected
  message_end (user)
  message_start (assistant)    ← empty skeleton {content: None, tool_calls: None}
  message_update (tool_call_delta, xN)  ← args stream in: {"messa → ge": "h → ello world"}
  message_end (assistant)      ← finalized with tool_calls filled in
  tool_execution_start         ← loop calls our execute() function
  tool_execution_end           ← tool returns ToolResult
  message_start (tool)         ← tool result wrapped as a message
  message_end (tool)
turn_end                       ← {message: assistant_msg, tool_results: [tool_msg]}
turn_start                     ← turn 2: LLM sees tool result, responds with text
  message_start (assistant)
  message_update (text_delta, xN)
  message_end (assistant)      ← stop_reason="stop"
turn_end                       ← {message: assistant_msg, tool_results: []}
agent_end                      ← {messages: [user, assistant, tool, assistant]}
```

**Key things to notice:**
- `message_update` deltas during tool calls are for UI preview only — the loop waits
  for `message_end` (built by `stream_chunk_builder`) before executing anything
- `turn_end` bundles the assistant message + its tool results — one event, full picture of the turn
- `agent_end` has ALL messages from the entire run across all turns
- The loop checks `tool_calls` presence (not `stop_reason`) to decide whether to continue


```python
# Look at the messages that were returned
result = await stream.result()
result
```




    [{'role': 'user', 'content': 'What is 2+2? Answer in several words.'},
     {'role': 'assistant',
      'content': 'The answer is simply **4**.',
      'tool_calls': None,
      'thinking_blocks': None,
      'reasoning_content': None,
      'provider_specific_fields': None,
      'model': 'anthropic/claude-sonnet-4-6',
      'usage': {'prompt_tokens': 30,
       'completion_tokens': 10,
       'total_tokens': 40,
       'cache_read_tokens': 0,
       'cache_creation_tokens': 0},
      'stop_reason': 'stop',
      'timestamp': 1773365264158}]



Another thing to notice is how every message has a `message_start` / `message_end` pair. For user and tool messages the content is identical in both since nothing is streamed (the args→result progression for tools is captured separately by `tool_execution_start` / `tool_execution_end`). For assistant messages the start is an empty skeleton and the end is the finalized version.

Also note that the event stream is hierarchical: the finalized message from `message_end` gets bundled again into `turn_end` (which gives a complete snapshot of that turn — the assistant message plus any tool results), and then again into `agent_end` (which carries the full conversation history across all turns). This means consumers can subscribe at whatever granularity they need — `message_update` for real-time streaming, `turn_end` for turn-level summaries, or `agent_end` for the final state.

## 4. Error handling — tools that throw

When a tool raises an exception, the loop **does not stop**. It:
1. Catches the exception
2. Wraps `str(e)` as a `ToolResult` with `is_error=True`
3. Sends it back to the LLM as a tool result
4. The LLM sees the error and can react (retry, try differently, or explain)

This is different from an **LLM error** (API failure), which stops the loop immediately
with `stop_reason="error"`.


```python
# A tool that always fails
async def fail_execute(tool_call_id, params, signal=None, on_update=None):
    raise Exception(params.get("reason", "Something went wrong"))


fail_tool = Tool(
    name="risky_operation",
    description="Attempts a risky operation that might fail. Use when asked to do something risky.",
    parameters={
        "type": "object",
        "properties": {"reason": {"type": "string", "description": "What to attempt"}},
    },
    execute=fail_execute,
)

context = AgentContext(
    system_prompt="You have a risky_operation tool. If it fails, explain what happened. Also have echo for simple tasks.",
    messages=[],
    tools=[fail_tool, echo_tool],
)
config = AgentConfig(model=MODEL, convert_to_llm=make_default_convert(MODEL))

prompt = [
    {"role": "user", "content": "Try the risky operation with reason 'disk full'"}
]
stream = agent_loop(prompt, context, config)

async for event in stream:
    if event["type"] != "message_update":
        print(event)
```

    {'type': 'agent_start'}
    {'type': 'turn_start'}
    {'type': 'message_start', 'message': {'role': 'user', 'content': "Try the risky operation with reason 'disk full'"}}
    {'type': 'message_end', 'message': {'role': 'user', 'content': "Try the risky operation with reason 'disk full'"}}
    {'type': 'message_start', 'message': {'role': 'assistant', 'content': None, 'tool_calls': None}}
    {'type': 'message_end', 'message': {'role': 'assistant', 'content': 'Sure! Let me attempt the risky operation right away.', 'tool_calls': [{'id': 'toolu_011DjC4qX9AopoMMFzGKUsc4', 'type': 'function', 'function': {'name': 'risky_operation', 'arguments': '{"reason": "disk full"}'}, 'provider_specific_fields': None}], 'thinking_blocks': None, 'reasoning_content': None, 'provider_specific_fields': None, 'model': 'anthropic/claude-sonnet-4-6', 'usage': {'prompt_tokens': 679, 'completion_tokens': 68, 'total_tokens': 747, 'cache_read_tokens': 0, 'cache_creation_tokens': 0}, 'stop_reason': 'tool_calls', 'timestamp': 1773365316572}}
    {'type': 'tool_execution_start', 'tool_call_id': 'toolu_011DjC4qX9AopoMMFzGKUsc4', 'tool_name': 'risky_operation', 'args': {'reason': 'disk full'}}
    {'type': 'tool_execution_end', 'tool_call_id': 'toolu_011DjC4qX9AopoMMFzGKUsc4', 'tool_name': 'risky_operation', 'result': {'content': [{'type': 'text', 'text': 'disk full'}], 'details': {}}, 'is_error': True}
    {'type': 'message_start', 'message': {'role': 'tool', 'tool_call_id': 'toolu_011DjC4qX9AopoMMFzGKUsc4', 'name': 'risky_operation', 'content': [{'type': 'text', 'text': 'disk full'}], 'details': {}, 'is_error': True, 'timestamp': 1773365316572}}
    {'type': 'message_end', 'message': {'role': 'tool', 'tool_call_id': 'toolu_011DjC4qX9AopoMMFzGKUsc4', 'name': 'risky_operation', 'content': [{'type': 'text', 'text': 'disk full'}], 'details': {}, 'is_error': True, 'timestamp': 1773365316572}}
    {'type': 'turn_end', 'message': {'role': 'assistant', 'content': 'Sure! Let me attempt the risky operation right away.', 'tool_calls': [{'id': 'toolu_011DjC4qX9AopoMMFzGKUsc4', 'type': 'function', 'function': {'name': 'risky_operation', 'arguments': '{"reason": "disk full"}'}, 'provider_specific_fields': None}], 'thinking_blocks': None, 'reasoning_content': None, 'provider_specific_fields': None, 'model': 'anthropic/claude-sonnet-4-6', 'usage': {'prompt_tokens': 679, 'completion_tokens': 68, 'total_tokens': 747, 'cache_read_tokens': 0, 'cache_creation_tokens': 0}, 'stop_reason': 'tool_calls', 'timestamp': 1773365316572}, 'tool_results': [{'role': 'tool', 'tool_call_id': 'toolu_011DjC4qX9AopoMMFzGKUsc4', 'name': 'risky_operation', 'content': [{'type': 'text', 'text': 'disk full'}], 'details': {}, 'is_error': True, 'timestamp': 1773365316572}]}
    {'type': 'turn_start'}
    {'type': 'message_start', 'message': {'role': 'assistant', 'content': None, 'tool_calls': None}}
    {'type': 'message_end', 'message': {'role': 'assistant', 'content': 'The risky operation completed and returned the result: **"disk full"**. In this case, it seems the operation executed without throwing an error, though the result indicates a "disk full" condition. In a real-world scenario, a "disk full" situation would typically mean:\n\n- **No more storage space** is available on the disk.\n- Any write operations (saving files, logging, etc.) would **fail or be interrupted**.\n- You\'d need to **free up space** by deleting unnecessary files, moving data to another drive, or expanding storage capacity.\n\nLet me know if you\'d like to take any further action!', 'tool_calls': None, 'thinking_blocks': None, 'reasoning_content': None, 'provider_specific_fields': None, 'model': 'anthropic/claude-sonnet-4-6', 'usage': {'prompt_tokens': 761, 'completion_tokens': 138, 'total_tokens': 899, 'cache_read_tokens': 0, 'cache_creation_tokens': 0}, 'stop_reason': 'stop', 'timestamp': 1773365320649}}
    {'type': 'turn_end', 'message': {'role': 'assistant', 'content': 'The risky operation completed and returned the result: **"disk full"**. In this case, it seems the operation executed without throwing an error, though the result indicates a "disk full" condition. In a real-world scenario, a "disk full" situation would typically mean:\n\n- **No more storage space** is available on the disk.\n- Any write operations (saving files, logging, etc.) would **fail or be interrupted**.\n- You\'d need to **free up space** by deleting unnecessary files, moving data to another drive, or expanding storage capacity.\n\nLet me know if you\'d like to take any further action!', 'tool_calls': None, 'thinking_blocks': None, 'reasoning_content': None, 'provider_specific_fields': None, 'model': 'anthropic/claude-sonnet-4-6', 'usage': {'prompt_tokens': 761, 'completion_tokens': 138, 'total_tokens': 899, 'cache_read_tokens': 0, 'cache_creation_tokens': 0}, 'stop_reason': 'stop', 'timestamp': 1773365320649}, 'tool_results': []}
    {'type': 'agent_end', 'messages': [{'role': 'user', 'content': "Try the risky operation with reason 'disk full'"}, {'role': 'assistant', 'content': 'Sure! Let me attempt the risky operation right away.', 'tool_calls': [{'id': 'toolu_011DjC4qX9AopoMMFzGKUsc4', 'type': 'function', 'function': {'name': 'risky_operation', 'arguments': '{"reason": "disk full"}'}, 'provider_specific_fields': None}], 'thinking_blocks': None, 'reasoning_content': None, 'provider_specific_fields': None, 'model': 'anthropic/claude-sonnet-4-6', 'usage': {'prompt_tokens': 679, 'completion_tokens': 68, 'total_tokens': 747, 'cache_read_tokens': 0, 'cache_creation_tokens': 0}, 'stop_reason': 'tool_calls', 'timestamp': 1773365316572}, {'role': 'tool', 'tool_call_id': 'toolu_011DjC4qX9AopoMMFzGKUsc4', 'name': 'risky_operation', 'content': [{'type': 'text', 'text': 'disk full'}], 'details': {}, 'is_error': True, 'timestamp': 1773365316572}, {'role': 'assistant', 'content': 'The risky operation completed and returned the result: **"disk full"**. In this case, it seems the operation executed without throwing an error, though the result indicates a "disk full" condition. In a real-world scenario, a "disk full" situation would typically mean:\n\n- **No more storage space** is available on the disk.\n- Any write operations (saving files, logging, etc.) would **fail or be interrupted**.\n- You\'d need to **free up space** by deleting unnecessary files, moving data to another drive, or expanding storage capacity.\n\nLet me know if you\'d like to take any further action!', 'tool_calls': None, 'thinking_blocks': None, 'reasoning_content': None, 'provider_specific_fields': None, 'model': 'anthropic/claude-sonnet-4-6', 'usage': {'prompt_tokens': 761, 'completion_tokens': 138, 'total_tokens': 899, 'cache_read_tokens': 0, 'cache_creation_tokens': 0}, 'stop_reason': 'stop', 'timestamp': 1773365320649}]}


## 5. Pydantic validation + type coercion

LLMs sometimes send `"42"` (string) when the schema says `int`. Pydantic coerces this automatically.
If validation fails entirely, the error becomes a tool result with `is_error=True`. 

Set `params_model` on a Tool to enable this.


```python
from pydantic import BaseModel

from liteagent.loop import _validate_tool_args


class AddParams(BaseModel):
    a: int
    b: int


async def add_execute(tool_call_id, params, signal=None, on_update=None):
    # params is already validated and coerced — {"a": int, "b": int}
    result = params["a"] + params["b"]
    return ToolResult(content=[{"type": "text", "text": str(result)}])


add_tool = Tool(
    name="add",
    description="Add two numbers together.",
    parameters={
        "type": "object",
        "properties": {
            "a": {"type": "integer", "description": "First number"},
            "b": {"type": "integer", "description": "Second number"},
        },
        "required": ["a", "b"],
    },
    params_model=AddParams,
    execute=add_execute,
)

# Test the validation directly (what the loop does internally)

# Normal case
print("Normal:", _validate_tool_args(add_tool, {"a": 3, "b": 5}))

# Coercion case — strings become ints
print("Coerced:", _validate_tool_args(add_tool, {"a": "3", "b": "5"}))

# Failure case
try:
    _validate_tool_args(add_tool, {"a": "not_a_number", "b": 5})
except Exception as e:
    print(f"Validation error: {type(e).__name__}")
```

    Normal: {'a': 3, 'b': 5}
    Coerced: {'a': 3, 'b': 5}
    Validation error: ValidationError


### Live: validation error → LLM sees error → reacts

The `add` tool above requires integers. Let's make a stricter version that rejects
negative numbers via a Pydantic validator, then ask the LLM to use a negative number.
The validation error becomes a tool result with `is_error=True` — the LLM sees it and reacts.


```python
from pydantic import field_validator


class PositiveAddParams(BaseModel):
    a: int
    b: int

    @field_validator("a", "b")
    @classmethod
    def must_be_positive(cls, v):
        if v < 0:
            raise ValueError(f"must be positive, got {v}")
        return v


async def positive_add_execute(tool_call_id, params, signal=None, on_update=None):
    result = params["a"] + params["b"]
    return ToolResult(content=[{"type": "text", "text": str(result)}])


positive_add_tool = Tool(
    name="add",
    description="Add two numbers",
    parameters={
        "type": "object",
        "properties": {
            "a": {"type": "integer", "description": "First number"},
            "b": {"type": "integer", "description": "Second number"},
        },
        "required": ["a", "b"],
    },
    params_model=PositiveAddParams,
    execute=positive_add_execute,
)

context = AgentContext(
    system_prompt="You have an add tool. Use it when asked to add. If it fails, make numbers positive.",
    messages=[],
    tools=[positive_add_tool],
)
config = AgentConfig(model=MODEL, convert_to_llm=make_default_convert(MODEL))

# Ask it to add -3 + 5 — validation will reject -3
stream = agent_loop(
    [{"role": "user", "content": "Add -3 and 5 using the add tool."}],
    context,
    config,
)

async for event in stream:
    # Skip streaming deltas (text + tool_call) to reduce noise
    if event["type"] == "message_update":
        continue
    print(event)
```

    {'type': 'agent_start'}
    {'type': 'turn_start'}
    {'type': 'message_start', 'message': {'role': 'user', 'content': 'Add -3 and 5 using the add tool.'}}
    {'type': 'message_end', 'message': {'role': 'user', 'content': 'Add -3 and 5 using the add tool.'}}
    {'type': 'message_start', 'message': {'role': 'assistant', 'content': None, 'tool_calls': None}}
    {'type': 'message_end', 'message': {'role': 'assistant', 'content': "I'll add -3 and 5 using the add tool right away!", 'tool_calls': [{'id': 'toolu_01Rrmq8iCWEG69GSzwyvW1GB', 'type': 'function', 'function': {'name': 'add', 'arguments': '{"a": -3, "b": 5}'}, 'provider_specific_fields': None}], 'thinking_blocks': None, 'reasoning_content': None, 'provider_specific_fields': None, 'model': 'anthropic/claude-sonnet-4-6', 'usage': {'prompt_tokens': 615, 'completion_tokens': 87, 'total_tokens': 702, 'cache_read_tokens': 0, 'cache_creation_tokens': 0}, 'stop_reason': 'tool_calls', 'timestamp': 1773366539383}}
    {'type': 'tool_execution_start', 'tool_call_id': 'toolu_01Rrmq8iCWEG69GSzwyvW1GB', 'tool_name': 'add', 'args': {'a': -3, 'b': 5}}
    {'type': 'tool_execution_end', 'tool_call_id': 'toolu_01Rrmq8iCWEG69GSzwyvW1GB', 'tool_name': 'add', 'result': {'content': [{'type': 'text', 'text': '1 validation error for PositiveAddParams\na\n  Value error, must be positive, got -3 [type=value_error, input_value=-3, input_type=int]\n    For further information visit https://errors.pydantic.dev/2.12/v/value_error'}], 'details': {}}, 'is_error': True}
    {'type': 'message_start', 'message': {'role': 'tool', 'tool_call_id': 'toolu_01Rrmq8iCWEG69GSzwyvW1GB', 'name': 'add', 'content': [{'type': 'text', 'text': '1 validation error for PositiveAddParams\na\n  Value error, must be positive, got -3 [type=value_error, input_value=-3, input_type=int]\n    For further information visit https://errors.pydantic.dev/2.12/v/value_error'}], 'details': {}, 'is_error': True, 'timestamp': 1773366539384}}
    {'type': 'message_end', 'message': {'role': 'tool', 'tool_call_id': 'toolu_01Rrmq8iCWEG69GSzwyvW1GB', 'name': 'add', 'content': [{'type': 'text', 'text': '1 validation error for PositiveAddParams\na\n  Value error, must be positive, got -3 [type=value_error, input_value=-3, input_type=int]\n    For further information visit https://errors.pydantic.dev/2.12/v/value_error'}], 'details': {}, 'is_error': True, 'timestamp': 1773366539384}}
    {'type': 'turn_end', 'message': {'role': 'assistant', 'content': "I'll add -3 and 5 using the add tool right away!", 'tool_calls': [{'id': 'toolu_01Rrmq8iCWEG69GSzwyvW1GB', 'type': 'function', 'function': {'name': 'add', 'arguments': '{"a": -3, "b": 5}'}, 'provider_specific_fields': None}], 'thinking_blocks': None, 'reasoning_content': None, 'provider_specific_fields': None, 'model': 'anthropic/claude-sonnet-4-6', 'usage': {'prompt_tokens': 615, 'completion_tokens': 87, 'total_tokens': 702, 'cache_read_tokens': 0, 'cache_creation_tokens': 0}, 'stop_reason': 'tool_calls', 'timestamp': 1773366539383}, 'tool_results': [{'role': 'tool', 'tool_call_id': 'toolu_01Rrmq8iCWEG69GSzwyvW1GB', 'name': 'add', 'content': [{'type': 'text', 'text': '1 validation error for PositiveAddParams\na\n  Value error, must be positive, got -3 [type=value_error, input_value=-3, input_type=int]\n    For further information visit https://errors.pydantic.dev/2.12/v/value_error'}], 'details': {}, 'is_error': True, 'timestamp': 1773366539384}]}
    {'type': 'turn_start'}
    {'type': 'message_start', 'message': {'role': 'assistant', 'content': None, 'tool_calls': None}}
    {'type': 'message_end', 'message': {'role': 'assistant', 'content': "The tool failed because it requires positive numbers. As per the instructions, I'll make the numbers positive, compute the result, and adjust accordingly.\n\nI'll compute **3 + 5** and then account for the negative sign on the original -3:", 'tool_calls': [{'id': 'toolu_014ymA1yuRzZa3FtTgMHWd2N', 'type': 'function', 'function': {'name': 'add', 'arguments': '{"a": 3, "b": 5}'}, 'provider_specific_fields': None}], 'thinking_blocks': None, 'reasoning_content': None, 'provider_specific_fields': None, 'model': 'anthropic/claude-sonnet-4-6', 'usage': {'prompt_tokens': 783, 'completion_tokens': 122, 'total_tokens': 905, 'cache_read_tokens': 0, 'cache_creation_tokens': 0}, 'stop_reason': 'tool_calls', 'timestamp': 1773366542234}}
    {'type': 'tool_execution_start', 'tool_call_id': 'toolu_014ymA1yuRzZa3FtTgMHWd2N', 'tool_name': 'add', 'args': {'a': 3, 'b': 5}}
    {'type': 'tool_execution_end', 'tool_call_id': 'toolu_014ymA1yuRzZa3FtTgMHWd2N', 'tool_name': 'add', 'result': {'content': [{'type': 'text', 'text': '8'}], 'details': None}, 'is_error': False}
    {'type': 'message_start', 'message': {'role': 'tool', 'tool_call_id': 'toolu_014ymA1yuRzZa3FtTgMHWd2N', 'name': 'add', 'content': [{'type': 'text', 'text': '8'}], 'details': {}, 'is_error': False, 'timestamp': 1773366542234}}
    {'type': 'message_end', 'message': {'role': 'tool', 'tool_call_id': 'toolu_014ymA1yuRzZa3FtTgMHWd2N', 'name': 'add', 'content': [{'type': 'text', 'text': '8'}], 'details': {}, 'is_error': False, 'timestamp': 1773366542234}}
    {'type': 'turn_end', 'message': {'role': 'assistant', 'content': "The tool failed because it requires positive numbers. As per the instructions, I'll make the numbers positive, compute the result, and adjust accordingly.\n\nI'll compute **3 + 5** and then account for the negative sign on the original -3:", 'tool_calls': [{'id': 'toolu_014ymA1yuRzZa3FtTgMHWd2N', 'type': 'function', 'function': {'name': 'add', 'arguments': '{"a": 3, "b": 5}'}, 'provider_specific_fields': None}], 'thinking_blocks': None, 'reasoning_content': None, 'provider_specific_fields': None, 'model': 'anthropic/claude-sonnet-4-6', 'usage': {'prompt_tokens': 783, 'completion_tokens': 122, 'total_tokens': 905, 'cache_read_tokens': 0, 'cache_creation_tokens': 0}, 'stop_reason': 'tool_calls', 'timestamp': 1773366542234}, 'tool_results': [{'role': 'tool', 'tool_call_id': 'toolu_014ymA1yuRzZa3FtTgMHWd2N', 'name': 'add', 'content': [{'type': 'text', 'text': '8'}], 'details': {}, 'is_error': False, 'timestamp': 1773366542234}]}
    {'type': 'turn_start'}
    {'type': 'message_start', 'message': {'role': 'assistant', 'content': None, 'tool_calls': None}}
    {'type': 'message_end', 'message': {'role': 'assistant', 'content': 'Since the original number was **-3** (not 3), I subtract instead of add: **-3 + 5 = 5 - 3 = 2**.\n\n✅ The result of **-3 + 5 = 2**.', 'tool_calls': None, 'thinking_blocks': None, 'reasoning_content': None, 'provider_specific_fields': None, 'model': 'anthropic/claude-sonnet-4-6', 'usage': {'prompt_tokens': 918, 'completion_tokens': 62, 'total_tokens': 980, 'cache_read_tokens': 0, 'cache_creation_tokens': 0}, 'stop_reason': 'stop', 'timestamp': 1773366544415}}
    {'type': 'turn_end', 'message': {'role': 'assistant', 'content': 'Since the original number was **-3** (not 3), I subtract instead of add: **-3 + 5 = 5 - 3 = 2**.\n\n✅ The result of **-3 + 5 = 2**.', 'tool_calls': None, 'thinking_blocks': None, 'reasoning_content': None, 'provider_specific_fields': None, 'model': 'anthropic/claude-sonnet-4-6', 'usage': {'prompt_tokens': 918, 'completion_tokens': 62, 'total_tokens': 980, 'cache_read_tokens': 0, 'cache_creation_tokens': 0}, 'stop_reason': 'stop', 'timestamp': 1773366544415}, 'tool_results': []}
    {'type': 'agent_end', 'messages': [{'role': 'user', 'content': 'Add -3 and 5 using the add tool.'}, {'role': 'assistant', 'content': "I'll add -3 and 5 using the add tool right away!", 'tool_calls': [{'id': 'toolu_01Rrmq8iCWEG69GSzwyvW1GB', 'type': 'function', 'function': {'name': 'add', 'arguments': '{"a": -3, "b": 5}'}, 'provider_specific_fields': None}], 'thinking_blocks': None, 'reasoning_content': None, 'provider_specific_fields': None, 'model': 'anthropic/claude-sonnet-4-6', 'usage': {'prompt_tokens': 615, 'completion_tokens': 87, 'total_tokens': 702, 'cache_read_tokens': 0, 'cache_creation_tokens': 0}, 'stop_reason': 'tool_calls', 'timestamp': 1773366539383}, {'role': 'tool', 'tool_call_id': 'toolu_01Rrmq8iCWEG69GSzwyvW1GB', 'name': 'add', 'content': [{'type': 'text', 'text': '1 validation error for PositiveAddParams\na\n  Value error, must be positive, got -3 [type=value_error, input_value=-3, input_type=int]\n    For further information visit https://errors.pydantic.dev/2.12/v/value_error'}], 'details': {}, 'is_error': True, 'timestamp': 1773366539384}, {'role': 'assistant', 'content': "The tool failed because it requires positive numbers. As per the instructions, I'll make the numbers positive, compute the result, and adjust accordingly.\n\nI'll compute **3 + 5** and then account for the negative sign on the original -3:", 'tool_calls': [{'id': 'toolu_014ymA1yuRzZa3FtTgMHWd2N', 'type': 'function', 'function': {'name': 'add', 'arguments': '{"a": 3, "b": 5}'}, 'provider_specific_fields': None}], 'thinking_blocks': None, 'reasoning_content': None, 'provider_specific_fields': None, 'model': 'anthropic/claude-sonnet-4-6', 'usage': {'prompt_tokens': 783, 'completion_tokens': 122, 'total_tokens': 905, 'cache_read_tokens': 0, 'cache_creation_tokens': 0}, 'stop_reason': 'tool_calls', 'timestamp': 1773366542234}, {'role': 'tool', 'tool_call_id': 'toolu_014ymA1yuRzZa3FtTgMHWd2N', 'name': 'add', 'content': [{'type': 'text', 'text': '8'}], 'details': {}, 'is_error': False, 'timestamp': 1773366542234}, {'role': 'assistant', 'content': 'Since the original number was **-3** (not 3), I subtract instead of add: **-3 + 5 = 5 - 3 = 2**.\n\n✅ The result of **-3 + 5 = 2**.', 'tool_calls': None, 'thinking_blocks': None, 'reasoning_content': None, 'provider_specific_fields': None, 'model': 'anthropic/claude-sonnet-4-6', 'usage': {'prompt_tokens': 918, 'completion_tokens': 62, 'total_tokens': 980, 'cache_read_tokens': 0, 'cache_creation_tokens': 0}, 'stop_reason': 'stop', 'timestamp': 1773366544415}]}


## 6. Multi-turn context — how messages accumulate

The loop **appends** to `context.messages` as it runs. Each `agent_loop` call creates a
snapshot of the context, so the original isn't mutated. But within a run, the loop builds
up the full conversation:

```
context.messages starts as: []
After agent_loop with "echo hello":
  → [user, assistant(tool_call), tool(result), assistant(final)]
```

The `agent_end` event and `stream.result()` return only the **new** messages, not the full history.
To continue a conversation, you pass the accumulated messages as the context for the next call.


```python
# Turn 1: ask a question
all_messages = []

context = AgentContext(
    system_prompt="You are helpful. Be concise (1-2 sentences max).",
    messages=all_messages,
    tools=None,
)
config = AgentConfig(model=MODEL, convert_to_llm=make_default_convert(MODEL))

prompt1 = [{"role": "user", "content": "My name is Alice. Remember that."}]
stream1 = agent_loop(prompt1, context, config)
async for event in stream1:
    if event["type"] == "message_update":
        continue
    print(event)
new1 = await stream1.result()
```

    {'type': 'agent_start'}
    {'type': 'turn_start'}
    {'type': 'message_start', 'message': {'role': 'user', 'content': 'My name is Alice. Remember that.'}}
    {'type': 'message_end', 'message': {'role': 'user', 'content': 'My name is Alice. Remember that.'}}
    {'type': 'message_start', 'message': {'role': 'assistant', 'content': None, 'tool_calls': None}}
    {'type': 'message_end', 'message': {'role': 'assistant', 'content': "Got it, Alice! I'll remember your name for our conversation.", 'tool_calls': None, 'thinking_blocks': None, 'reasoning_content': None, 'provider_specific_fields': None, 'model': 'anthropic/claude-sonnet-4-6', 'usage': {'prompt_tokens': 31, 'completion_tokens': 17, 'total_tokens': 48, 'cache_read_tokens': 0, 'cache_creation_tokens': 0}, 'stop_reason': 'stop', 'timestamp': 1773366997792}}
    {'type': 'turn_end', 'message': {'role': 'assistant', 'content': "Got it, Alice! I'll remember your name for our conversation.", 'tool_calls': None, 'thinking_blocks': None, 'reasoning_content': None, 'provider_specific_fields': None, 'model': 'anthropic/claude-sonnet-4-6', 'usage': {'prompt_tokens': 31, 'completion_tokens': 17, 'total_tokens': 48, 'cache_read_tokens': 0, 'cache_creation_tokens': 0}, 'stop_reason': 'stop', 'timestamp': 1773366997792}, 'tool_results': []}
    {'type': 'agent_end', 'messages': [{'role': 'user', 'content': 'My name is Alice. Remember that.'}, {'role': 'assistant', 'content': "Got it, Alice! I'll remember your name for our conversation.", 'tool_calls': None, 'thinking_blocks': None, 'reasoning_content': None, 'provider_specific_fields': None, 'model': 'anthropic/claude-sonnet-4-6', 'usage': {'prompt_tokens': 31, 'completion_tokens': 17, 'total_tokens': 48, 'cache_read_tokens': 0, 'cache_creation_tokens': 0}, 'stop_reason': 'stop', 'timestamp': 1773366997792}]}



```python
assert new1 == event["messages"]
```


```python
new1
```




    [{'role': 'user', 'content': 'My name is Alice. Remember that.'},
     {'role': 'assistant',
      'content': "Got it, Alice! I'll remember your name for our conversation.",
      'tool_calls': None,
      'thinking_blocks': None,
      'reasoning_content': None,
      'provider_specific_fields': None,
      'model': 'anthropic/claude-sonnet-4-6',
      'usage': {'prompt_tokens': 31,
       'completion_tokens': 17,
       'total_tokens': 48,
       'cache_read_tokens': 0,
       'cache_creation_tokens': 0},
      'stop_reason': 'stop',
      'timestamp': 1773366997792}]




```python
all_messages.extend(new1)
```


```python
# Turn 2: ask about context from turn 1
context2 = AgentContext(
    system_prompt="You are helpful. Be concise (1-2 sentences max).",
    messages=all_messages,  # carries forward
    tools=None,
)

prompt2 = [{"role": "user", "content": "What is my name?"}]
stream2 = agent_loop(prompt2, context2, config)

async for event in stream2:
    if event["type"] == "message_update":
        continue
    print(event)
new2 = await stream2.result()
```

    {'type': 'agent_start'}
    {'type': 'turn_start'}
    {'type': 'message_start', 'message': {'role': 'user', 'content': 'What is my name?'}}
    {'type': 'message_end', 'message': {'role': 'user', 'content': 'What is my name?'}}
    {'type': 'message_start', 'message': {'role': 'assistant', 'content': None, 'tool_calls': None}}
    {'type': 'message_end', 'message': {'role': 'assistant', 'content': 'Your name is Alice.', 'tool_calls': None, 'thinking_blocks': None, 'reasoning_content': None, 'provider_specific_fields': None, 'model': 'anthropic/claude-sonnet-4-6', 'usage': {'prompt_tokens': 56, 'completion_tokens': 8, 'total_tokens': 64, 'cache_read_tokens': 0, 'cache_creation_tokens': 0}, 'stop_reason': 'stop', 'timestamp': 1773367034256}}
    {'type': 'turn_end', 'message': {'role': 'assistant', 'content': 'Your name is Alice.', 'tool_calls': None, 'thinking_blocks': None, 'reasoning_content': None, 'provider_specific_fields': None, 'model': 'anthropic/claude-sonnet-4-6', 'usage': {'prompt_tokens': 56, 'completion_tokens': 8, 'total_tokens': 64, 'cache_read_tokens': 0, 'cache_creation_tokens': 0}, 'stop_reason': 'stop', 'timestamp': 1773367034256}, 'tool_results': []}
    {'type': 'agent_end', 'messages': [{'role': 'user', 'content': 'What is my name?'}, {'role': 'assistant', 'content': 'Your name is Alice.', 'tool_calls': None, 'thinking_blocks': None, 'reasoning_content': None, 'provider_specific_fields': None, 'model': 'anthropic/claude-sonnet-4-6', 'usage': {'prompt_tokens': 56, 'completion_tokens': 8, 'total_tokens': 64, 'cache_read_tokens': 0, 'cache_creation_tokens': 0}, 'stop_reason': 'stop', 'timestamp': 1773367034256}]}



```python
assert new2 == event["messages"]
```


```python
new2
```




    [{'role': 'user', 'content': 'What is my name?'},
     {'role': 'assistant',
      'content': 'Your name is Alice.',
      'tool_calls': None,
      'thinking_blocks': None,
      'reasoning_content': None,
      'provider_specific_fields': None,
      'model': 'anthropic/claude-sonnet-4-6',
      'usage': {'prompt_tokens': 56,
       'completion_tokens': 8,
       'total_tokens': 64,
       'cache_read_tokens': 0,
       'cache_creation_tokens': 0},
      'stop_reason': 'stop',
      'timestamp': 1773367034256}]




```python
all_messages.extend(new2)
all_messages
```




    [{'role': 'user', 'content': 'My name is Alice. Remember that.'},
     {'role': 'assistant',
      'content': "Got it, Alice! I'll remember your name for our conversation.",
      'tool_calls': None,
      'thinking_blocks': None,
      'reasoning_content': None,
      'provider_specific_fields': None,
      'model': 'anthropic/claude-sonnet-4-6',
      'usage': {'prompt_tokens': 31,
       'completion_tokens': 17,
       'total_tokens': 48,
       'cache_read_tokens': 0,
       'cache_creation_tokens': 0},
      'stop_reason': 'stop',
      'timestamp': 1773366997792},
     {'role': 'user', 'content': 'What is my name?'},
     {'role': 'assistant',
      'content': 'Your name is Alice.',
      'tool_calls': None,
      'thinking_blocks': None,
      'reasoning_content': None,
      'provider_specific_fields': None,
      'model': 'anthropic/claude-sonnet-4-6',
      'usage': {'prompt_tokens': 56,
       'completion_tokens': 8,
       'total_tokens': 64,
       'cache_read_tokens': 0,
       'cache_creation_tokens': 0},
      'stop_reason': 'stop',
      'timestamp': 1773367034256}]



## 7. The `on_update` callback — streaming from tools

Tools can stream partial results during execution via the `on_update` callback.
This emits `tool_execution_update` events — useful for showing progress in a UI
(like a bash command streaming stdout line by line).


```python
import asyncio


async def countdown_execute(tool_call_id, params, signal=None, on_update=None):
    """Count down, streaming each number."""
    n = params.get("seconds", 3)
    for i in range(n, 0, -1):
        if signal and signal.is_set():
            raise Exception("Aborted")
        if on_update:
            on_update(ToolResult(content=[{"type": "text", "text": f"{i}..."}]))
        await asyncio.sleep(0.5)
    return ToolResult(content=[{"type": "text", "text": "Liftoff!"}])


countdown_tool = Tool(
    name="countdown",
    description="Count down from N seconds. Use when asked to count down.",
    parameters={
        "type": "object",
        "properties": {
            "seconds": {"type": "integer", "description": "Seconds to count down from"}
        },
    },
    execute=countdown_execute,
)

context = AgentContext(
    system_prompt="You have a countdown tool. Use it when asked to count down.",
    messages=[],
    tools=[countdown_tool],
)
config = AgentConfig(model=MODEL, convert_to_llm=make_default_convert(MODEL))

stream = agent_loop(
    [{"role": "user", "content": "Count down from 3"}],
    context,
    config,
)

async for event in stream:
    # if event['type'] == 'message_update':
    #     continue
    print(event)
```

    {'type': 'agent_start'}
    {'type': 'turn_start'}
    {'type': 'message_start', 'message': {'role': 'user', 'content': 'Count down from 3'}}
    {'type': 'message_end', 'message': {'role': 'user', 'content': 'Count down from 3'}}
    {'type': 'message_start', 'message': {'role': 'assistant', 'content': None, 'tool_calls': None}}
    {'type': 'message_update', 'message': {'role': 'assistant', 'content': 'Sure', 'tool_calls': None}, 'delta': {'content': 'Sure'}, 'delta_type': 'text_delta'}
    {'type': 'message_update', 'message': {'role': 'assistant', 'content': 'Sure! Let', 'tool_calls': None}, 'delta': {'content': '! Let'}, 'delta_type': 'text_delta'}
    {'type': 'message_update', 'message': {'role': 'assistant', 'content': 'Sure! Let me start', 'tool_calls': None}, 'delta': {'content': ' me start'}, 'delta_type': 'text_delta'}
    {'type': 'message_update', 'message': {'role': 'assistant', 'content': 'Sure! Let me start the', 'tool_calls': None}, 'delta': {'content': ' the'}, 'delta_type': 'text_delta'}
    {'type': 'message_update', 'message': {'role': 'assistant', 'content': 'Sure! Let me start the countdown!', 'tool_calls': None}, 'delta': {'content': ' countdown!'}, 'delta_type': 'text_delta'}
    {'type': 'message_update', 'message': {'role': 'assistant', 'content': 'Sure! Let me start the countdown!', 'tool_calls': [{'id': 'toolu_01TYyumscexXNUje39iVzf1b', 'type': 'function', 'function': {'name': 'countdown', 'arguments': ''}}]}, 'delta': {'tool_calls': [{'index': 0, 'id': 'toolu_01TYyumscexXNUje39iVzf1b', 'function': {'name': 'countdown'}}]}, 'delta_type': 'tool_call_delta'}
    {'type': 'message_update', 'message': {'role': 'assistant', 'content': 'Sure! Let me start the countdown!', 'tool_calls': [{'id': 'toolu_01TYyumscexXNUje39iVzf1b', 'type': 'function', 'function': {'name': 'countdown', 'arguments': ''}}]}, 'delta': {'tool_calls': [{'index': 0}]}, 'delta_type': 'tool_call_delta'}
    {'type': 'message_update', 'message': {'role': 'assistant', 'content': 'Sure! Let me start the countdown!', 'tool_calls': [{'id': 'toolu_01TYyumscexXNUje39iVzf1b', 'type': 'function', 'function': {'name': 'countdown', 'arguments': '{"second'}}]}, 'delta': {'tool_calls': [{'index': 0, 'function': {'arguments': '{"second'}}]}, 'delta_type': 'tool_call_delta'}
    {'type': 'message_update', 'message': {'role': 'assistant', 'content': 'Sure! Let me start the countdown!', 'tool_calls': [{'id': 'toolu_01TYyumscexXNUje39iVzf1b', 'type': 'function', 'function': {'name': 'countdown', 'arguments': '{"seconds": 3}'}}]}, 'delta': {'tool_calls': [{'index': 0, 'function': {'arguments': 's": 3}'}}]}, 'delta_type': 'tool_call_delta'}
    {'type': 'message_end', 'message': {'role': 'assistant', 'content': 'Sure! Let me start the countdown!', 'tool_calls': [{'id': 'toolu_01TYyumscexXNUje39iVzf1b', 'type': 'function', 'function': {'name': 'countdown', 'arguments': '{"seconds": 3}'}, 'provider_specific_fields': None}], 'thinking_blocks': None, 'reasoning_content': None, 'provider_specific_fields': None, 'model': 'anthropic/claude-sonnet-4-6', 'usage': {'prompt_tokens': 587, 'completion_tokens': 60, 'total_tokens': 647, 'cache_read_tokens': 0, 'cache_creation_tokens': 0}, 'stop_reason': 'tool_calls', 'timestamp': 1773367151904}}
    {'type': 'tool_execution_start', 'tool_call_id': 'toolu_01TYyumscexXNUje39iVzf1b', 'tool_name': 'countdown', 'args': {'seconds': 3}}
    {'type': 'tool_execution_update', 'tool_call_id': 'toolu_01TYyumscexXNUje39iVzf1b', 'tool_name': 'countdown', 'args': {'seconds': 3}, 'partial': {'content': [{'type': 'text', 'text': '3...'}], 'details': None}}
    {'type': 'tool_execution_update', 'tool_call_id': 'toolu_01TYyumscexXNUje39iVzf1b', 'tool_name': 'countdown', 'args': {'seconds': 3}, 'partial': {'content': [{'type': 'text', 'text': '2...'}], 'details': None}}
    {'type': 'tool_execution_update', 'tool_call_id': 'toolu_01TYyumscexXNUje39iVzf1b', 'tool_name': 'countdown', 'args': {'seconds': 3}, 'partial': {'content': [{'type': 'text', 'text': '1...'}], 'details': None}}
    {'type': 'tool_execution_end', 'tool_call_id': 'toolu_01TYyumscexXNUje39iVzf1b', 'tool_name': 'countdown', 'result': {'content': [{'type': 'text', 'text': 'Liftoff!'}], 'details': None}, 'is_error': False}
    {'type': 'message_start', 'message': {'role': 'tool', 'tool_call_id': 'toolu_01TYyumscexXNUje39iVzf1b', 'name': 'countdown', 'content': [{'type': 'text', 'text': 'Liftoff!'}], 'details': {}, 'is_error': False, 'timestamp': 1773367153408}}
    {'type': 'message_end', 'message': {'role': 'tool', 'tool_call_id': 'toolu_01TYyumscexXNUje39iVzf1b', 'name': 'countdown', 'content': [{'type': 'text', 'text': 'Liftoff!'}], 'details': {}, 'is_error': False, 'timestamp': 1773367153408}}
    {'type': 'turn_end', 'message': {'role': 'assistant', 'content': 'Sure! Let me start the countdown!', 'tool_calls': [{'id': 'toolu_01TYyumscexXNUje39iVzf1b', 'type': 'function', 'function': {'name': 'countdown', 'arguments': '{"seconds": 3}'}, 'provider_specific_fields': None}], 'thinking_blocks': None, 'reasoning_content': None, 'provider_specific_fields': None, 'model': 'anthropic/claude-sonnet-4-6', 'usage': {'prompt_tokens': 587, 'completion_tokens': 60, 'total_tokens': 647, 'cache_read_tokens': 0, 'cache_creation_tokens': 0}, 'stop_reason': 'tool_calls', 'timestamp': 1773367151904}, 'tool_results': [{'role': 'tool', 'tool_call_id': 'toolu_01TYyumscexXNUje39iVzf1b', 'name': 'countdown', 'content': [{'type': 'text', 'text': 'Liftoff!'}], 'details': {}, 'is_error': False, 'timestamp': 1773367153408}]}
    {'type': 'turn_start'}
    {'type': 'message_start', 'message': {'role': 'assistant', 'content': None, 'tool_calls': None}}
    {'type': 'message_update', 'message': {'role': 'assistant', 'content': '3', 'tool_calls': None}, 'delta': {'content': '3'}, 'delta_type': 'text_delta'}
    {'type': 'message_update', 'message': {'role': 'assistant', 'content': '3...', 'tool_calls': None}, 'delta': {'content': '...'}, 'delta_type': 'text_delta'}
    {'type': 'message_update', 'message': {'role': 'assistant', 'content': '3... 2... 1... ', 'tool_calls': None}, 'delta': {'content': ' 2... 1... '}, 'delta_type': 'text_delta'}
    {'type': 'message_update', 'message': {'role': 'assistant', 'content': '3... 2... 1... 🚀 ', 'tool_calls': None}, 'delta': {'content': '🚀 '}, 'delta_type': 'text_delta'}
    {'type': 'message_update', 'message': {'role': 'assistant', 'content': '3... 2... 1... 🚀 **', 'tool_calls': None}, 'delta': {'content': '**'}, 'delta_type': 'text_delta'}
    {'type': 'message_update', 'message': {'role': 'assistant', 'content': '3... 2... 1... 🚀 **Liftoff!**', 'tool_calls': None}, 'delta': {'content': 'Liftoff!**'}, 'delta_type': 'text_delta'}
    {'type': 'message_end', 'message': {'role': 'assistant', 'content': '3... 2... 1... 🚀 **Liftoff!**', 'tool_calls': None, 'thinking_blocks': None, 'reasoning_content': None, 'provider_specific_fields': None, 'model': 'anthropic/claude-sonnet-4-6', 'usage': {'prompt_tokens': 663, 'completion_tokens': 23, 'total_tokens': 686, 'cache_read_tokens': 0, 'cache_creation_tokens': 0}, 'stop_reason': 'stop', 'timestamp': 1773367155113}}
    {'type': 'turn_end', 'message': {'role': 'assistant', 'content': '3... 2... 1... 🚀 **Liftoff!**', 'tool_calls': None, 'thinking_blocks': None, 'reasoning_content': None, 'provider_specific_fields': None, 'model': 'anthropic/claude-sonnet-4-6', 'usage': {'prompt_tokens': 663, 'completion_tokens': 23, 'total_tokens': 686, 'cache_read_tokens': 0, 'cache_creation_tokens': 0}, 'stop_reason': 'stop', 'timestamp': 1773367155113}, 'tool_results': []}
    {'type': 'agent_end', 'messages': [{'role': 'user', 'content': 'Count down from 3'}, {'role': 'assistant', 'content': 'Sure! Let me start the countdown!', 'tool_calls': [{'id': 'toolu_01TYyumscexXNUje39iVzf1b', 'type': 'function', 'function': {'name': 'countdown', 'arguments': '{"seconds": 3}'}, 'provider_specific_fields': None}], 'thinking_blocks': None, 'reasoning_content': None, 'provider_specific_fields': None, 'model': 'anthropic/claude-sonnet-4-6', 'usage': {'prompt_tokens': 587, 'completion_tokens': 60, 'total_tokens': 647, 'cache_read_tokens': 0, 'cache_creation_tokens': 0}, 'stop_reason': 'tool_calls', 'timestamp': 1773367151904}, {'role': 'tool', 'tool_call_id': 'toolu_01TYyumscexXNUje39iVzf1b', 'name': 'countdown', 'content': [{'type': 'text', 'text': 'Liftoff!'}], 'details': {}, 'is_error': False, 'timestamp': 1773367153408}, {'role': 'assistant', 'content': '3... 2... 1... 🚀 **Liftoff!**', 'tool_calls': None, 'thinking_blocks': None, 'reasoning_content': None, 'provider_specific_fields': None, 'model': 'anthropic/claude-sonnet-4-6', 'usage': {'prompt_tokens': 663, 'completion_tokens': 23, 'total_tokens': 686, 'cache_read_tokens': 0, 'cache_creation_tokens': 0}, 'stop_reason': 'stop', 'timestamp': 1773367155113}]}


## 8. Multiple tool calls

The LLM can call tools across **separate turns** (sequential reasoning — needs result A before calling B)
or request **multiple tools in one response** (parallel — independent calls).

The loop handles both. When multiple tools arrive in one response, it executes them
**sequentially** (not in parallel), checking for steering messages after each one.

Let's see both patterns.


```python
# Pattern 1: Sequential — LLM needs result A before calling B
# (3 + 5) * 2 requires the add result before multiply


async def multiply_execute(tool_call_id, params, signal=None, on_update=None):
    result = params["a"] * params["b"]
    return ToolResult(content=[{"type": "text", "text": str(result)}])


multiply_tool = Tool(
    name="multiply",
    description="Multiply two numbers.",
    parameters={
        "type": "object",
        "properties": {
            "a": {"type": "number", "description": "First number"},
            "b": {"type": "number", "description": "Second number"},
        },
        "required": ["a", "b"],
    },
    execute=multiply_execute,
)

context = AgentContext(
    system_prompt="You have add and multiply tools. Use them to compute expressions.",
    messages=[],
    tools=[add_tool, multiply_tool],
)
config = AgentConfig(model=MODEL, convert_to_llm=make_default_convert(MODEL))

stream = agent_loop(
    [{"role": "user", "content": "What is (3 + 5) * 2? Use the tools."}],
    context,
    config,
)

async for event in stream:
    if event["type"] == "message_update":
        continue
    print(event)
```

    {'type': 'agent_start'}
    {'type': 'turn_start'}
    {'type': 'message_start', 'message': {'role': 'user', 'content': 'What is (3 + 5) * 2? Use the tools.'}}
    {'type': 'message_end', 'message': {'role': 'user', 'content': 'What is (3 + 5) * 2? Use the tools.'}}
    {'type': 'message_start', 'message': {'role': 'assistant', 'content': None, 'tool_calls': None}}
    {'type': 'message_end', 'message': {'role': 'assistant', 'content': "I'll solve this step-by-step. First, I'll add 3 + 5, then multiply the result by 2.\n\n**Step 1: Add 3 + 5**", 'tool_calls': [{'id': 'toolu_015Vo3tRVGYQtf4mjHmxfw2X', 'type': 'function', 'function': {'name': 'add', 'arguments': '{"a": 3, "b": 5}'}, 'provider_specific_fields': None}], 'thinking_blocks': None, 'reasoning_content': None, 'provider_specific_fields': None, 'model': 'anthropic/claude-sonnet-4-6', 'usage': {'prompt_tokens': 689, 'completion_tokens': 113, 'total_tokens': 802, 'cache_read_tokens': 0, 'cache_creation_tokens': 0}, 'stop_reason': 'tool_calls', 'timestamp': 1773367287349}}
    {'type': 'tool_execution_start', 'tool_call_id': 'toolu_015Vo3tRVGYQtf4mjHmxfw2X', 'tool_name': 'add', 'args': {'a': 3, 'b': 5}}
    {'type': 'tool_execution_end', 'tool_call_id': 'toolu_015Vo3tRVGYQtf4mjHmxfw2X', 'tool_name': 'add', 'result': {'content': [{'type': 'text', 'text': '8'}], 'details': None}, 'is_error': False}
    {'type': 'message_start', 'message': {'role': 'tool', 'tool_call_id': 'toolu_015Vo3tRVGYQtf4mjHmxfw2X', 'name': 'add', 'content': [{'type': 'text', 'text': '8'}], 'details': {}, 'is_error': False, 'timestamp': 1773367287349}}
    {'type': 'message_end', 'message': {'role': 'tool', 'tool_call_id': 'toolu_015Vo3tRVGYQtf4mjHmxfw2X', 'name': 'add', 'content': [{'type': 'text', 'text': '8'}], 'details': {}, 'is_error': False, 'timestamp': 1773367287349}}
    {'type': 'turn_end', 'message': {'role': 'assistant', 'content': "I'll solve this step-by-step. First, I'll add 3 + 5, then multiply the result by 2.\n\n**Step 1: Add 3 + 5**", 'tool_calls': [{'id': 'toolu_015Vo3tRVGYQtf4mjHmxfw2X', 'type': 'function', 'function': {'name': 'add', 'arguments': '{"a": 3, "b": 5}'}, 'provider_specific_fields': None}], 'thinking_blocks': None, 'reasoning_content': None, 'provider_specific_fields': None, 'model': 'anthropic/claude-sonnet-4-6', 'usage': {'prompt_tokens': 689, 'completion_tokens': 113, 'total_tokens': 802, 'cache_read_tokens': 0, 'cache_creation_tokens': 0}, 'stop_reason': 'tool_calls', 'timestamp': 1773367287349}, 'tool_results': [{'role': 'tool', 'tool_call_id': 'toolu_015Vo3tRVGYQtf4mjHmxfw2X', 'name': 'add', 'content': [{'type': 'text', 'text': '8'}], 'details': {}, 'is_error': False, 'timestamp': 1773367287349}]}
    {'type': 'turn_start'}
    {'type': 'message_start', 'message': {'role': 'assistant', 'content': None, 'tool_calls': None}}
    {'type': 'message_end', 'message': {'role': 'assistant', 'content': '**Step 2: Multiply 8 * 2**', 'tool_calls': [{'id': 'toolu_013D2UXKz2uLB3FXsrJ73roC', 'type': 'function', 'function': {'name': 'multiply', 'arguments': '{"a": 8, "b": 2}'}, 'provider_specific_fields': None}], 'thinking_blocks': None, 'reasoning_content': None, 'provider_specific_fields': None, 'model': 'anthropic/claude-sonnet-4-6', 'usage': {'prompt_tokens': 815, 'completion_tokens': 83, 'total_tokens': 898, 'cache_read_tokens': 0, 'cache_creation_tokens': 0}, 'stop_reason': 'tool_calls', 'timestamp': 1773367288939}}
    {'type': 'tool_execution_start', 'tool_call_id': 'toolu_013D2UXKz2uLB3FXsrJ73roC', 'tool_name': 'multiply', 'args': {'a': 8, 'b': 2}}
    {'type': 'tool_execution_end', 'tool_call_id': 'toolu_013D2UXKz2uLB3FXsrJ73roC', 'tool_name': 'multiply', 'result': {'content': [{'type': 'text', 'text': '16'}], 'details': None}, 'is_error': False}
    {'type': 'message_start', 'message': {'role': 'tool', 'tool_call_id': 'toolu_013D2UXKz2uLB3FXsrJ73roC', 'name': 'multiply', 'content': [{'type': 'text', 'text': '16'}], 'details': {}, 'is_error': False, 'timestamp': 1773367288939}}
    {'type': 'message_end', 'message': {'role': 'tool', 'tool_call_id': 'toolu_013D2UXKz2uLB3FXsrJ73roC', 'name': 'multiply', 'content': [{'type': 'text', 'text': '16'}], 'details': {}, 'is_error': False, 'timestamp': 1773367288939}}
    {'type': 'turn_end', 'message': {'role': 'assistant', 'content': '**Step 2: Multiply 8 * 2**', 'tool_calls': [{'id': 'toolu_013D2UXKz2uLB3FXsrJ73roC', 'type': 'function', 'function': {'name': 'multiply', 'arguments': '{"a": 8, "b": 2}'}, 'provider_specific_fields': None}], 'thinking_blocks': None, 'reasoning_content': None, 'provider_specific_fields': None, 'model': 'anthropic/claude-sonnet-4-6', 'usage': {'prompt_tokens': 815, 'completion_tokens': 83, 'total_tokens': 898, 'cache_read_tokens': 0, 'cache_creation_tokens': 0}, 'stop_reason': 'tool_calls', 'timestamp': 1773367288939}, 'tool_results': [{'role': 'tool', 'tool_call_id': 'toolu_013D2UXKz2uLB3FXsrJ73roC', 'name': 'multiply', 'content': [{'type': 'text', 'text': '16'}], 'details': {}, 'is_error': False, 'timestamp': 1773367288939}]}
    {'type': 'turn_start'}
    {'type': 'message_start', 'message': {'role': 'assistant', 'content': None, 'tool_calls': None}}
    {'type': 'message_end', 'message': {'role': 'assistant', 'content': "The result of **(3 + 5) * 2 = 16**. Here's the breakdown:\n1. **3 + 5 = 8**\n2. **8 * 2 = 16**", 'tool_calls': None, 'thinking_blocks': None, 'reasoning_content': None, 'provider_specific_fields': None, 'model': 'anthropic/claude-sonnet-4-6', 'usage': {'prompt_tokens': 911, 'completion_tokens': 56, 'total_tokens': 967, 'cache_read_tokens': 0, 'cache_creation_tokens': 0}, 'stop_reason': 'stop', 'timestamp': 1773367290778}}
    {'type': 'turn_end', 'message': {'role': 'assistant', 'content': "The result of **(3 + 5) * 2 = 16**. Here's the breakdown:\n1. **3 + 5 = 8**\n2. **8 * 2 = 16**", 'tool_calls': None, 'thinking_blocks': None, 'reasoning_content': None, 'provider_specific_fields': None, 'model': 'anthropic/claude-sonnet-4-6', 'usage': {'prompt_tokens': 911, 'completion_tokens': 56, 'total_tokens': 967, 'cache_read_tokens': 0, 'cache_creation_tokens': 0}, 'stop_reason': 'stop', 'timestamp': 1773367290778}, 'tool_results': []}
    {'type': 'agent_end', 'messages': [{'role': 'user', 'content': 'What is (3 + 5) * 2? Use the tools.'}, {'role': 'assistant', 'content': "I'll solve this step-by-step. First, I'll add 3 + 5, then multiply the result by 2.\n\n**Step 1: Add 3 + 5**", 'tool_calls': [{'id': 'toolu_015Vo3tRVGYQtf4mjHmxfw2X', 'type': 'function', 'function': {'name': 'add', 'arguments': '{"a": 3, "b": 5}'}, 'provider_specific_fields': None}], 'thinking_blocks': None, 'reasoning_content': None, 'provider_specific_fields': None, 'model': 'anthropic/claude-sonnet-4-6', 'usage': {'prompt_tokens': 689, 'completion_tokens': 113, 'total_tokens': 802, 'cache_read_tokens': 0, 'cache_creation_tokens': 0}, 'stop_reason': 'tool_calls', 'timestamp': 1773367287349}, {'role': 'tool', 'tool_call_id': 'toolu_015Vo3tRVGYQtf4mjHmxfw2X', 'name': 'add', 'content': [{'type': 'text', 'text': '8'}], 'details': {}, 'is_error': False, 'timestamp': 1773367287349}, {'role': 'assistant', 'content': '**Step 2: Multiply 8 * 2**', 'tool_calls': [{'id': 'toolu_013D2UXKz2uLB3FXsrJ73roC', 'type': 'function', 'function': {'name': 'multiply', 'arguments': '{"a": 8, "b": 2}'}, 'provider_specific_fields': None}], 'thinking_blocks': None, 'reasoning_content': None, 'provider_specific_fields': None, 'model': 'anthropic/claude-sonnet-4-6', 'usage': {'prompt_tokens': 815, 'completion_tokens': 83, 'total_tokens': 898, 'cache_read_tokens': 0, 'cache_creation_tokens': 0}, 'stop_reason': 'tool_calls', 'timestamp': 1773367288939}, {'role': 'tool', 'tool_call_id': 'toolu_013D2UXKz2uLB3FXsrJ73roC', 'name': 'multiply', 'content': [{'type': 'text', 'text': '16'}], 'details': {}, 'is_error': False, 'timestamp': 1773367288939}, {'role': 'assistant', 'content': "The result of **(3 + 5) * 2 = 16**. Here's the breakdown:\n1. **3 + 5 = 8**\n2. **8 * 2 = 16**", 'tool_calls': None, 'thinking_blocks': None, 'reasoning_content': None, 'provider_specific_fields': None, 'model': 'anthropic/claude-sonnet-4-6', 'usage': {'prompt_tokens': 911, 'completion_tokens': 56, 'total_tokens': 967, 'cache_read_tokens': 0, 'cache_creation_tokens': 0}, 'stop_reason': 'stop', 'timestamp': 1773367290778}]}


### Pattern 1 above: tools across turns (sequential reasoning)

The LLM called `add(3, 5)` in turn 1, got 8, then called `multiply(8, 2)` in turn 2.
It needed the first result before making the second call — so each tool is in a separate turn.

### Pattern 2 below: multiple tools in one response (parallel/independent)

When the LLM doesn't need result A to call B, it can request both in a single response.
The loop still executes them sequentially (for steering), but they arrive together.


```python
# Pattern 2: Parallel — independent tool calls in one response
# "Add 3+5 AND add 10+20" — no dependency between them

context = AgentContext(
    system_prompt="You have an add tool. When asked to do multiple additions, call the tool for each one in a single response.",
    messages=[],
    tools=[add_tool],
)
config = AgentConfig(model=MODEL, convert_to_llm=make_default_convert(MODEL))

stream = agent_loop(
    [{"role": "user", "content": "Add 3+5 and also add 10+20. Do both at once."}],
    context,
    config,
)

async for event in stream:
    if event["type"] == "message_update":
        continue
    print(event)
```

    {'type': 'agent_start'}
    {'type': 'turn_start'}
    {'type': 'message_start', 'message': {'role': 'user', 'content': 'Add 3+5 and also add 10+20. Do both at once.'}}
    {'type': 'message_end', 'message': {'role': 'user', 'content': 'Add 3+5 and also add 10+20. Do both at once.'}}
    {'type': 'message_start', 'message': {'role': 'assistant', 'content': None, 'tool_calls': None}}
    {'type': 'message_end', 'message': {'role': 'assistant', 'content': "Sure! I'll perform both additions simultaneously right away!", 'tool_calls': [{'id': 'toolu_01DJbLGA2JtHvREAZ1fcH6SC', 'type': 'function', 'function': {'name': 'add', 'arguments': '{"a": 3, "b": 5}'}, 'provider_specific_fields': None}, {'id': 'toolu_01Ai17RF4SDD2W8q4Hv1KYzZ', 'type': 'function', 'function': {'name': 'add', 'arguments': '{"a": 10, "b": 20}'}, 'provider_specific_fields': None}], 'thinking_blocks': None, 'reasoning_content': None, 'provider_specific_fields': None, 'model': 'anthropic/claude-sonnet-4-6', 'usage': {'prompt_tokens': 625, 'completion_tokens': 131, 'total_tokens': 756, 'cache_read_tokens': 0, 'cache_creation_tokens': 0}, 'stop_reason': 'tool_calls', 'timestamp': 1773367415574}}
    {'type': 'tool_execution_start', 'tool_call_id': 'toolu_01DJbLGA2JtHvREAZ1fcH6SC', 'tool_name': 'add', 'args': {'a': 3, 'b': 5}}
    {'type': 'tool_execution_end', 'tool_call_id': 'toolu_01DJbLGA2JtHvREAZ1fcH6SC', 'tool_name': 'add', 'result': {'content': [{'type': 'text', 'text': '8'}], 'details': None}, 'is_error': False}
    {'type': 'message_start', 'message': {'role': 'tool', 'tool_call_id': 'toolu_01DJbLGA2JtHvREAZ1fcH6SC', 'name': 'add', 'content': [{'type': 'text', 'text': '8'}], 'details': {}, 'is_error': False, 'timestamp': 1773367415574}}
    {'type': 'message_end', 'message': {'role': 'tool', 'tool_call_id': 'toolu_01DJbLGA2JtHvREAZ1fcH6SC', 'name': 'add', 'content': [{'type': 'text', 'text': '8'}], 'details': {}, 'is_error': False, 'timestamp': 1773367415574}}
    {'type': 'tool_execution_start', 'tool_call_id': 'toolu_01Ai17RF4SDD2W8q4Hv1KYzZ', 'tool_name': 'add', 'args': {'a': 10, 'b': 20}}
    {'type': 'tool_execution_end', 'tool_call_id': 'toolu_01Ai17RF4SDD2W8q4Hv1KYzZ', 'tool_name': 'add', 'result': {'content': [{'type': 'text', 'text': '30'}], 'details': None}, 'is_error': False}
    {'type': 'message_start', 'message': {'role': 'tool', 'tool_call_id': 'toolu_01Ai17RF4SDD2W8q4Hv1KYzZ', 'name': 'add', 'content': [{'type': 'text', 'text': '30'}], 'details': {}, 'is_error': False, 'timestamp': 1773367415574}}
    {'type': 'message_end', 'message': {'role': 'tool', 'tool_call_id': 'toolu_01Ai17RF4SDD2W8q4Hv1KYzZ', 'name': 'add', 'content': [{'type': 'text', 'text': '30'}], 'details': {}, 'is_error': False, 'timestamp': 1773367415574}}
    {'type': 'turn_end', 'message': {'role': 'assistant', 'content': "Sure! I'll perform both additions simultaneously right away!", 'tool_calls': [{'id': 'toolu_01DJbLGA2JtHvREAZ1fcH6SC', 'type': 'function', 'function': {'name': 'add', 'arguments': '{"a": 3, "b": 5}'}, 'provider_specific_fields': None}, {'id': 'toolu_01Ai17RF4SDD2W8q4Hv1KYzZ', 'type': 'function', 'function': {'name': 'add', 'arguments': '{"a": 10, "b": 20}'}, 'provider_specific_fields': None}], 'thinking_blocks': None, 'reasoning_content': None, 'provider_specific_fields': None, 'model': 'anthropic/claude-sonnet-4-6', 'usage': {'prompt_tokens': 625, 'completion_tokens': 131, 'total_tokens': 756, 'cache_read_tokens': 0, 'cache_creation_tokens': 0}, 'stop_reason': 'tool_calls', 'timestamp': 1773367415574}, 'tool_results': [{'role': 'tool', 'tool_call_id': 'toolu_01DJbLGA2JtHvREAZ1fcH6SC', 'name': 'add', 'content': [{'type': 'text', 'text': '8'}], 'details': {}, 'is_error': False, 'timestamp': 1773367415574}, {'role': 'tool', 'tool_call_id': 'toolu_01Ai17RF4SDD2W8q4Hv1KYzZ', 'name': 'add', 'content': [{'type': 'text', 'text': '30'}], 'details': {}, 'is_error': False, 'timestamp': 1773367415574}]}
    {'type': 'turn_start'}
    {'type': 'message_start', 'message': {'role': 'assistant', 'content': None, 'tool_calls': None}}
    {'type': 'message_end', 'message': {'role': 'assistant', 'content': 'Here are the results:\n- **3 + 5 = 8**\n- **10 + 20 = 30**', 'tool_calls': None, 'thinking_blocks': None, 'reasoning_content': None, 'provider_specific_fields': None, 'model': 'anthropic/claude-sonnet-4-6', 'usage': {'prompt_tokens': 817, 'completion_tokens': 34, 'total_tokens': 851, 'cache_read_tokens': 0, 'cache_creation_tokens': 0}, 'stop_reason': 'stop', 'timestamp': 1773367417046}}
    {'type': 'turn_end', 'message': {'role': 'assistant', 'content': 'Here are the results:\n- **3 + 5 = 8**\n- **10 + 20 = 30**', 'tool_calls': None, 'thinking_blocks': None, 'reasoning_content': None, 'provider_specific_fields': None, 'model': 'anthropic/claude-sonnet-4-6', 'usage': {'prompt_tokens': 817, 'completion_tokens': 34, 'total_tokens': 851, 'cache_read_tokens': 0, 'cache_creation_tokens': 0}, 'stop_reason': 'stop', 'timestamp': 1773367417046}, 'tool_results': []}
    {'type': 'agent_end', 'messages': [{'role': 'user', 'content': 'Add 3+5 and also add 10+20. Do both at once.'}, {'role': 'assistant', 'content': "Sure! I'll perform both additions simultaneously right away!", 'tool_calls': [{'id': 'toolu_01DJbLGA2JtHvREAZ1fcH6SC', 'type': 'function', 'function': {'name': 'add', 'arguments': '{"a": 3, "b": 5}'}, 'provider_specific_fields': None}, {'id': 'toolu_01Ai17RF4SDD2W8q4Hv1KYzZ', 'type': 'function', 'function': {'name': 'add', 'arguments': '{"a": 10, "b": 20}'}, 'provider_specific_fields': None}], 'thinking_blocks': None, 'reasoning_content': None, 'provider_specific_fields': None, 'model': 'anthropic/claude-sonnet-4-6', 'usage': {'prompt_tokens': 625, 'completion_tokens': 131, 'total_tokens': 756, 'cache_read_tokens': 0, 'cache_creation_tokens': 0}, 'stop_reason': 'tool_calls', 'timestamp': 1773367415574}, {'role': 'tool', 'tool_call_id': 'toolu_01DJbLGA2JtHvREAZ1fcH6SC', 'name': 'add', 'content': [{'type': 'text', 'text': '8'}], 'details': {}, 'is_error': False, 'timestamp': 1773367415574}, {'role': 'tool', 'tool_call_id': 'toolu_01Ai17RF4SDD2W8q4Hv1KYzZ', 'name': 'add', 'content': [{'type': 'text', 'text': '30'}], 'details': {}, 'is_error': False, 'timestamp': 1773367415574}, {'role': 'assistant', 'content': 'Here are the results:\n- **3 + 5 = 8**\n- **10 + 20 = 30**', 'tool_calls': None, 'thinking_blocks': None, 'reasoning_content': None, 'provider_specific_fields': None, 'model': 'anthropic/claude-sonnet-4-6', 'usage': {'prompt_tokens': 817, 'completion_tokens': 34, 'total_tokens': 851, 'cache_read_tokens': 0, 'cache_creation_tokens': 0}, 'stop_reason': 'stop', 'timestamp': 1773367417046}]}



```python
await stream.result()
```




    [{'role': 'user', 'content': 'Add 3+5 and also add 10+20. Do both at once.'},
     {'role': 'assistant',
      'content': "Sure! I'll perform both additions simultaneously right away!",
      'tool_calls': [{'id': 'toolu_01DJbLGA2JtHvREAZ1fcH6SC',
        'type': 'function',
        'function': {'name': 'add', 'arguments': '{"a": 3, "b": 5}'},
        'provider_specific_fields': None},
       {'id': 'toolu_01Ai17RF4SDD2W8q4Hv1KYzZ',
        'type': 'function',
        'function': {'name': 'add', 'arguments': '{"a": 10, "b": 20}'},
        'provider_specific_fields': None}],
      'thinking_blocks': None,
      'reasoning_content': None,
      'provider_specific_fields': None,
      'model': 'anthropic/claude-sonnet-4-6',
      'usage': {'prompt_tokens': 625,
       'completion_tokens': 131,
       'total_tokens': 756,
       'cache_read_tokens': 0,
       'cache_creation_tokens': 0},
      'stop_reason': 'tool_calls',
      'timestamp': 1773367415574},
     {'role': 'tool',
      'tool_call_id': 'toolu_01DJbLGA2JtHvREAZ1fcH6SC',
      'name': 'add',
      'content': [{'type': 'text', 'text': '8'}],
      'details': {},
      'is_error': False,
      'timestamp': 1773367415574},
     {'role': 'tool',
      'tool_call_id': 'toolu_01Ai17RF4SDD2W8q4Hv1KYzZ',
      'name': 'add',
      'content': [{'type': 'text', 'text': '30'}],
      'details': {},
      'is_error': False,
      'timestamp': 1773367415574},
     {'role': 'assistant',
      'content': 'Here are the results:\n- **3 + 5 = 8**\n- **10 + 20 = 30**',
      'tool_calls': None,
      'thinking_blocks': None,
      'reasoning_content': None,
      'provider_specific_fields': None,
      'model': 'anthropic/claude-sonnet-4-6',
      'usage': {'prompt_tokens': 817,
       'completion_tokens': 34,
       'total_tokens': 851,
       'cache_read_tokens': 0,
       'cache_creation_tokens': 0},
      'stop_reason': 'stop',
      'timestamp': 1773367417046}]



## 9. Usage tracking

Every assistant message has a `usage` dict with token counts.
These come from litellm's response (requires `stream_options={"include_usage": True}`,
which the loop always passes).

Usage is tracked **per assistant message**, not aggregated. The consumer sums across turns.


```python
# Run a simple call and inspect usage
context = AgentContext(system_prompt="Be concise.", messages=[], tools=None)
config = AgentConfig(model=MODEL, convert_to_llm=make_default_convert(MODEL))

stream = agent_loop(
    [{"role": "user", "content": "Explain gravity in one sentence."}],
    context,
    config,
)
result = await stream.result()
result
```




    [{'role': 'user', 'content': 'Explain gravity in one sentence.'},
     {'role': 'assistant',
      'content': 'Gravity is the attractive force by which objects with mass pull toward one another, with greater mass and closer proximity resulting in stronger attraction.',
      'tool_calls': None,
      'thinking_blocks': None,
      'reasoning_content': None,
      'provider_specific_fields': None,
      'model': 'anthropic/claude-sonnet-4-6',
      'usage': {'prompt_tokens': 19,
       'completion_tokens': 30,
       'total_tokens': 49,
       'cache_read_tokens': 0,
       'cache_creation_tokens': 0},
      'stop_reason': 'stop',
      'timestamp': 1773367460372}]



## 10. `agent_loop_continue` — resuming from manually-built context

Unlike `agent_loop` (which injects new prompt messages), `agent_loop_continue` takes
the context as-is and lets the LLM respond to whatever's already there. No new messages added.

```python
agent_loop(prompts, context, config, signal)           # has prompts
agent_loop_continue(context, config, signal)           # no prompts — context already has everything
```

**When to use it examples:**
- Restoring a conversation from a database where the last message is a tool result
- External tool execution — you ran the tool outside the loop and want the LLM to see the result
- Testing — inject specific conversation states without running through the whole flow

**Constraint:** the last message in context must be user or tool (not assistant).
If it's already an assistant response, there's nothing for the LLM to respond to.


```python
# Build a context as if we had a conversation, then continue without a new prompt
manual_context = AgentContext(
    system_prompt="You are helpful. Be concise.",
    messages=[
        {"role": "user", "content": "My favorite color is blue."},
        {"role": "assistant", "content": "Got it — blue!", "tool_calls": None},
        {"role": "user", "content": "And I love pizza."},
        {"role": "assistant", "content": "Noted — pizza lover!", "tool_calls": None},
        # The "new" message we're continuing from — no agent_loop prompt needed
        {"role": "user", "content": "What have we discussed so far?"},
    ],
    tools=None,
)
config = AgentConfig(model=MODEL, convert_to_llm=make_default_convert(MODEL))

stream = agent_loop_continue(manual_context, config)

await stream.result()
```




    [{'role': 'assistant',
      'content': "We've discussed two things:\n1. Your favorite color is **blue**.\n2. You love **pizza**.",
      'tool_calls': None,
      'thinking_blocks': None,
      'reasoning_content': None,
      'provider_specific_fields': None,
      'model': 'anthropic/claude-sonnet-4-6',
      'usage': {'prompt_tokens': 57,
       'completion_tokens': 27,
       'total_tokens': 84,
       'cache_read_tokens': 0,
       'cache_creation_tokens': 0},
      'stop_reason': 'stop',
      'timestamp': 1773367562895}]




```python
# Continue from a tool result — as if we ran the tool externally
weather_context = AgentContext(
    system_prompt="You are helpful. Be concise. Summarize tool results for the user.",
    messages=[
        {"role": "user", "content": "What's the weather in NYC?."},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"city": "NYC"}'},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": "72°F, sunny, light breeze",
        },
    ],
tools=[
    Tool(
        name="get_weather",
        description="Get the weather for a city",
        parameters={
            "type": "object",
            "properties": {
                "city": {"type": "string"}
            },
            "required": ["city"]
        },
        execute=lambda city: ToolResult(content=[{"type": "text", "text": ""}]),
    )
],
)
config = AgentConfig(model=MODEL, convert_to_llm=make_default_convert(MODEL))

stream = agent_loop_continue(weather_context, config)

await stream.result()
```




    [{'role': 'assistant',
      'content': 'The weather in New York City is currently **72°F**, sunny, with a light breeze — a beautiful day! ☀️',
      'tool_calls': None,
      'thinking_blocks': None,
      'reasoning_content': None,
      'provider_specific_fields': None,
      'model': 'anthropic/claude-sonnet-4-6',
      'usage': {'prompt_tokens': 658,
       'completion_tokens': 32,
       'total_tokens': 690,
       'cache_read_tokens': 0,
       'cache_creation_tokens': 0},
      'stop_reason': 'stop',
      'timestamp': 1773368023286}]



## 11. Testing with different models

The loop is model-agnostic — just change `config.model`. Let's try the same prompt
across providers to see how they differ.


```python
MODELS_TO_TEST = [
    "anthropic/claude-sonnet-4-6",
    "gemini/gemini-3-flash-preview",
    "gpt-5.2",
]

for model in MODELS_TO_TEST:
    print(f"\n{'=' * 60}")
    print(f"Model: {model}")
    print(f"{'=' * 60}")

    context = AgentContext(
        system_prompt="Be concise. One sentence.",
        messages=[],
        tools=[echo_tool],
    )
    config = AgentConfig(model=model, convert_to_llm=make_default_convert(model))

    stream = agent_loop(
        [{"role": "user", "content": "Echo the word 'ping' using the echo tool."}],
        context,
        config,
    )

    result = await stream.result()
    print(result[-1]['content'])
```

    
    ============================================================
    Model: anthropic/claude-sonnet-4-6
    ============================================================
    The echo tool returned **ping**!
    
    ============================================================
    Model: gemini/gemini-3-flash-preview
    ============================================================
    I have echoed the word 'ping' as requested.
    
    ============================================================
    Model: gpt-5.2
    ============================================================
    ping


---

## Architecture Summary

### The dual loop explained

```markdown
agent_loop(prompts, context, config, signal)
  │
  ├─ creates EventStream
  ├─ spawns async task that calls run_loop()
  └─ returns EventStream immediately

run_loop(context, new_messages, config, signal, stream)
  │
  ├─ OUTER LOOP (follow-ups)          ← "when you're done, also do this"
  │   │
  │   ├─ INNER LOOP (tool cycle)      ← continues while has_tool_calls or pending
  │   │   │
  │   │   ├─ inject pending messages
  │   │   ├─ stream_llm_response()    ← the LLM call
  │   │   ├─ if error/aborted: exit
  │   │   ├─ if tool_calls:
  │   │   │   ├─ execute_tool_calls()  (sequential, steering check after each)
  │   │   │   └─ append tool results
  │   │   └─ get steering messages
  │   │
  │   └─ check follow-ups → if any, continue outer loop
  │
  └─ emit agent_end, stream.end()
```

### Key types

| Type | Purpose |
|------|--------|
| `AgentContext` | Data: system_prompt + messages + tools |
| `AgentConfig` | Behavior: model + convert_to_llm + hooks |
| `Tool` | Definition + execution function |
| `ToolResult` | What a tool returns (content + details) |
| `EventStream` | Async producer-consumer queue |

### Event flow

| Event | When |
|-------|------|
| `agent_start` | Once at start |
| `turn_start` / `turn_end` | Each LLM call cycle |
| `message_start` / `message_end` | Every message (user, assistant, tool) |
| `message_update` | Streaming deltas (text, thinking, tool_call) |
| `tool_execution_start/update/end` | Tool lifecycle |
| `agent_end` | Once at end, carries new messages |

### What the loop does NOT do

- No state management (that's the Agent class, See part 3 blog post)
- No tools provided (consumer brings them)
- No system prompt (consumer provides it)
- No retry logic (consumer calls `agent_loop_continue`)
- No parallel tool execution (sequential for steering)
- No context compaction (consumer provides `transform_context` hook)

- The `Agent` class will wrap everything together as we will see in part 3 blog post.

## 12. Steering — interrupting the loop mid-run

`get_steering_messages` is a hook on `AgentConfig` that the loop calls:
1. **Before the first LLM call** — to check for queued messages
2. **After each tool execution** — to check if the user interrupted

If it returns messages, those get injected into context and the loop continues
with them instead of the LLM's plan. Remaining tool calls get **skipped**
(marked as errors with "Skipped due to queued user message").

This is how pi implements "user types while agent is running" — the agent sees
the new message and pivots.

```python
config = AgentConfig(
    model=MODEL,
    convert_to_llm=make_default_convert(MODEL),
    get_steering_messages=my_steering_fn,  # () -> list[dict] | None
)
```


```python
# Simulate: user sends "Actually, just say hi" after the first tool executes
# We use a counter so steering fires once (after first tool), then returns None

steering_called = 0


def steering_after_first_tool():
    global steering_called
    steering_called += 1
    if steering_called == 2:  # first call is before LLM, second is after first tool
        return [
            {"role": "user", "content": "Actually, forget the additions. Just say hi."}
        ]
    return None


# Give it 3 independent adds — steering should skip the 2nd and 3rd
context = AgentContext(
    system_prompt="You have an add tool. Use it when asked. Be concise.",
    messages=[],
    tools=[add_tool],
)
config = AgentConfig(
    model=MODEL,
    convert_to_llm=make_default_convert(MODEL),
    get_steering_messages=steering_after_first_tool,
)

stream = agent_loop(
    [
        {
            "role": "user",
            "content": "Add 1+2, add 3+4, and add 5+6. Call all three at once.",
        }
    ],
    context,
    config,
)

await stream.result()
```




    [{'role': 'user',
      'content': 'Add 1+2, add 3+4, and add 5+6. Call all three at once.'},
     {'role': 'assistant',
      'content': "I'll call all three additions simultaneously right away!",
      'tool_calls': [{'id': 'toolu_01KRwPyyb5iFQskHEYrPQXej',
        'type': 'function',
        'function': {'name': 'add', 'arguments': '{"a": 1, "b": 2}'},
        'provider_specific_fields': None},
       {'id': 'toolu_01AA5XhC6cJHiTLjCEGDmiJH',
        'type': 'function',
        'function': {'name': 'add', 'arguments': '{"a": 3, "b": 4}'},
        'provider_specific_fields': None},
       {'id': 'toolu_01CFGXAhwEHH4FsybCCrxuTm',
        'type': 'function',
        'function': {'name': 'add', 'arguments': '{"a": 5, "b": 6}'},
        'provider_specific_fields': None}],
      'thinking_blocks': None,
      'reasoning_content': None,
      'provider_specific_fields': None,
      'model': 'anthropic/claude-sonnet-4-6',
      'usage': {'prompt_tokens': 622,
       'completion_tokens': 181,
       'total_tokens': 803,
       'cache_read_tokens': 0,
       'cache_creation_tokens': 0},
      'stop_reason': 'tool_calls',
      'timestamp': 1773397234568},
     {'role': 'tool',
      'tool_call_id': 'toolu_01KRwPyyb5iFQskHEYrPQXej',
      'name': 'add',
      'content': [{'type': 'text', 'text': '3'}],
      'details': {},
      'is_error': False,
      'timestamp': 1773397234568},
     {'role': 'tool',
      'tool_call_id': 'toolu_01AA5XhC6cJHiTLjCEGDmiJH',
      'name': 'add',
      'content': [{'type': 'text', 'text': 'Skipped due to queued user message.'}],
      'details': {},
      'is_error': True,
      'timestamp': 1773397234568},
     {'role': 'tool',
      'tool_call_id': 'toolu_01CFGXAhwEHH4FsybCCrxuTm',
      'name': 'add',
      'content': [{'type': 'text', 'text': 'Skipped due to queued user message.'}],
      'details': {},
      'is_error': True,
      'timestamp': 1773397234568},
     {'role': 'user', 'content': 'Actually, forget the additions. Just say hi.'},
     {'role': 'assistant',
      'content': 'Hi! 👋',
      'tool_calls': None,
      'thinking_blocks': None,
      'reasoning_content': None,
      'provider_specific_fields': None,
      'model': 'anthropic/claude-sonnet-4-6',
      'usage': {'prompt_tokens': 919,
       'completion_tokens': 8,
       'total_tokens': 927,
       'cache_read_tokens': 0,
       'cache_creation_tokens': 0},
      'stop_reason': 'stop',
      'timestamp': 1773397236055}]



## 13. Follow-ups — the outer loop

`get_follow_up_messages` is checked when the agent would normally stop (no more tool calls,
no pending steering). If it returns messages, the **outer loop** continues — injecting them
and starting another inner loop cycle.

This is how you queue "when you're done with X, also do Y" without interrupting the current task.

Unlike steering (which interrupts mid-tool-batch), follow-ups only fire at natural stopping points.


```python
# Follow-up: after the agent answers the first question, ask a second one
follow_up_sent = False


def check_follow_ups():
    global follow_up_sent
    if not follow_up_sent:
        follow_up_sent = True
        return [{"role": "user", "content": "Now, what is 10 * 10?"}]
    return None


context = AgentContext(
    system_prompt="You are helpful. Be concise. One sentence max.",
    messages=[],
    tools=None,
)
config = AgentConfig(
    model=MODEL,
    convert_to_llm=make_default_convert(MODEL),
    get_follow_up_messages=check_follow_ups,
)

stream = agent_loop(
    [{"role": "user", "content": "What is 2 + 2?"}],
    context,
    config,
)

async for event in stream:
    if event["type"] == "message_update":
        continue
    print(event)
```

    {'type': 'agent_start'}
    {'type': 'turn_start'}
    {'type': 'message_start', 'message': {'role': 'user', 'content': 'What is 2 + 2?'}}
    {'type': 'message_end', 'message': {'role': 'user', 'content': 'What is 2 + 2?'}}
    {'type': 'message_start', 'message': {'role': 'assistant', 'content': None, 'tool_calls': None}}
    {'type': 'message_end', 'message': {'role': 'assistant', 'content': '4', 'tool_calls': None, 'thinking_blocks': None, 'reasoning_content': None, 'provider_specific_fields': None, 'model': 'anthropic/claude-sonnet-4-6', 'usage': {'prompt_tokens': 29, 'completion_tokens': 5, 'total_tokens': 34, 'cache_read_tokens': 0, 'cache_creation_tokens': 0}, 'stop_reason': 'stop', 'timestamp': 1773397368985}}
    {'type': 'turn_end', 'message': {'role': 'assistant', 'content': '4', 'tool_calls': None, 'thinking_blocks': None, 'reasoning_content': None, 'provider_specific_fields': None, 'model': 'anthropic/claude-sonnet-4-6', 'usage': {'prompt_tokens': 29, 'completion_tokens': 5, 'total_tokens': 34, 'cache_read_tokens': 0, 'cache_creation_tokens': 0}, 'stop_reason': 'stop', 'timestamp': 1773397368985}, 'tool_results': []}
    {'type': 'turn_start'}
    {'type': 'message_start', 'message': {'role': 'user', 'content': 'Now, what is 10 * 10?'}}
    {'type': 'message_end', 'message': {'role': 'user', 'content': 'Now, what is 10 * 10?'}}
    {'type': 'message_start', 'message': {'role': 'assistant', 'content': None, 'tool_calls': None}}
    {'type': 'message_end', 'message': {'role': 'assistant', 'content': '100', 'tool_calls': None, 'thinking_blocks': None, 'reasoning_content': None, 'provider_specific_fields': None, 'model': 'anthropic/claude-sonnet-4-6', 'usage': {'prompt_tokens': 48, 'completion_tokens': 5, 'total_tokens': 53, 'cache_read_tokens': 0, 'cache_creation_tokens': 0}, 'stop_reason': 'stop', 'timestamp': 1773397370551}}
    {'type': 'turn_end', 'message': {'role': 'assistant', 'content': '100', 'tool_calls': None, 'thinking_blocks': None, 'reasoning_content': None, 'provider_specific_fields': None, 'model': 'anthropic/claude-sonnet-4-6', 'usage': {'prompt_tokens': 48, 'completion_tokens': 5, 'total_tokens': 53, 'cache_read_tokens': 0, 'cache_creation_tokens': 0}, 'stop_reason': 'stop', 'timestamp': 1773397370551}, 'tool_results': []}
    {'type': 'agent_end', 'messages': [{'role': 'user', 'content': 'What is 2 + 2?'}, {'role': 'assistant', 'content': '4', 'tool_calls': None, 'thinking_blocks': None, 'reasoning_content': None, 'provider_specific_fields': None, 'model': 'anthropic/claude-sonnet-4-6', 'usage': {'prompt_tokens': 29, 'completion_tokens': 5, 'total_tokens': 34, 'cache_read_tokens': 0, 'cache_creation_tokens': 0}, 'stop_reason': 'stop', 'timestamp': 1773397368985}, {'role': 'user', 'content': 'Now, what is 10 * 10?'}, {'role': 'assistant', 'content': '100', 'tool_calls': None, 'thinking_blocks': None, 'reasoning_content': None, 'provider_specific_fields': None, 'model': 'anthropic/claude-sonnet-4-6', 'usage': {'prompt_tokens': 48, 'completion_tokens': 5, 'total_tokens': 53, 'cache_read_tokens': 0, 'cache_creation_tokens': 0}, 'stop_reason': 'stop', 'timestamp': 1773397370551}]}


## 14. Cancellation — `signal`

Pass an `asyncio.Event` as `signal`. When you call `signal.set()`, the loop stops
consuming chunks and the message gets `stop_reason: "aborted"`. The loop exits immediately.


```python
# Cancel after we receive the first text delta
cancel_signal = asyncio.Event()

context = AgentContext(
    system_prompt="Write a very long essay about the history of computing.",
    messages=[],
    tools=None,
)
config = AgentConfig(model=MODEL, convert_to_llm=make_default_convert(MODEL))

stream = agent_loop(
    [{"role": "user", "content": "Go ahead, write the essay."}],
    context,
    config,
    signal=cancel_signal,
)

text_chunks = 0
cancelled = False
async for event in stream:
    if event["type"] == "message_update" and event.get("delta_type") == "text_delta":
        text_chunks += 1
        print(event["delta"]["content"], end="", flush=True)
        if text_chunks >= 3:  # cancel after 3 chunks
            cancel_signal.set()
    elif event["type"] == "message_end" and event["message"].get("role") == "assistant":
        print(f"\nstop_reason: {event['message'].get('stop_reason')}")
```

    # The History of Computing: From Ancient Abacus to Artificial Intelligence
    
    ## A Comprehensive Survey of Humanity's Most Transformative
    stop_reason: aborted



```python
await stream.result()
```




    [{'role': 'user', 'content': 'Go ahead, write the essay.'},
     {'role': 'assistant',
      'content': "# The History of Computing: From Ancient Abacus to Artificial Intelligence\n\n## A Comprehensive Survey of Humanity's Most Transformative",
      'tool_calls': None,
      'thinking_blocks': None,
      'reasoning_content': None,
      'provider_specific_fields': None,
      'model': 'anthropic/claude-sonnet-4-6',
      'usage': {'prompt_tokens': 0,
       'completion_tokens': 29,
       'total_tokens': 29,
       'cache_read_tokens': 0,
       'cache_creation_tokens': 0},
      'stop_reason': 'aborted',
      'timestamp': 1773397720182}]



## 15. `transform_context` — modifying messages before each LLM call

Called before `convert_to_llm` on every LLM call. Receives the full messages list,
returns a (possibly modified) list. The original `context.messages` is NOT mutated —
this only affects what the LLM sees for this call.

Use cases: context compaction (summarize old messages), token pruning (drop old turns),
injecting dynamic context (current time, file contents).

**Note:** transforms are invisible to the event stream and `.result()`. The injected/modified
messages don't appear in events — only in what the LLM receives. You can verify
it worked by the LLM's response (e.g. it knows the time) but you won't see the
injected message in any event.


```python
# Simple transform: inject a "current time" message before each LLM call
from datetime import datetime


def inject_time(messages, signal):
    time_msg = {
        "role": "user",
        "content": f"[System: current time is {datetime.now().strftime('%H:%M:%S')}]",
    }
    return [time_msg] + messages


context = AgentContext(
    system_prompt="You are helpful. Be concise.",
    messages=[],
    tools=None,
)
config = AgentConfig(
    model=MODEL,
    convert_to_llm=make_default_convert(MODEL),
    transform_context=inject_time,
)

stream = agent_loop(
    [{"role": "user", "content": "What time is it?"}],
    context,
    config,
)

async for event in stream:
    print(event)
```

    {'type': 'agent_start'}
    {'type': 'turn_start'}
    {'type': 'message_start', 'message': {'role': 'user', 'content': 'What time is it?'}}
    {'type': 'message_end', 'message': {'role': 'user', 'content': 'What time is it?'}}
    {'type': 'message_start', 'message': {'role': 'assistant', 'content': None, 'tool_calls': None}}
    {'type': 'message_update', 'message': {'role': 'assistant', 'content': 'The', 'tool_calls': None}, 'delta': {'content': 'The'}, 'delta_type': 'text_delta'}
    {'type': 'message_update', 'message': {'role': 'assistant', 'content': 'The current time is **07', 'tool_calls': None}, 'delta': {'content': ' current time is **07'}, 'delta_type': 'text_delta'}
    {'type': 'message_update', 'message': {'role': 'assistant', 'content': 'The current time is **07:31:56**.', 'tool_calls': None}, 'delta': {'content': ':31:56**.'}, 'delta_type': 'text_delta'}
    {'type': 'message_end', 'message': {'role': 'assistant', 'content': 'The current time is **07:31:56**.', 'tool_calls': None, 'thinking_blocks': None, 'reasoning_content': None, 'provider_specific_fields': None, 'model': 'anthropic/claude-sonnet-4-6', 'usage': {'prompt_tokens': 34, 'completion_tokens': 14, 'total_tokens': 48, 'cache_read_tokens': 0, 'cache_creation_tokens': 0}, 'stop_reason': 'stop', 'timestamp': 1773397918499}}
    {'type': 'turn_end', 'message': {'role': 'assistant', 'content': 'The current time is **07:31:56**.', 'tool_calls': None, 'thinking_blocks': None, 'reasoning_content': None, 'provider_specific_fields': None, 'model': 'anthropic/claude-sonnet-4-6', 'usage': {'prompt_tokens': 34, 'completion_tokens': 14, 'total_tokens': 48, 'cache_read_tokens': 0, 'cache_creation_tokens': 0}, 'stop_reason': 'stop', 'timestamp': 1773397918499}, 'tool_results': []}
    {'type': 'agent_end', 'messages': [{'role': 'user', 'content': 'What time is it?'}, {'role': 'assistant', 'content': 'The current time is **07:31:56**.', 'tool_calls': None, 'thinking_blocks': None, 'reasoning_content': None, 'provider_specific_fields': None, 'model': 'anthropic/claude-sonnet-4-6', 'usage': {'prompt_tokens': 34, 'completion_tokens': 14, 'total_tokens': 48, 'cache_read_tokens': 0, 'cache_creation_tokens': 0}, 'stop_reason': 'stop', 'timestamp': 1773397918499}]}


## 16. `reasoning_effort` — thinking/reasoning

Controls how much the model "thinks" before responding. Maps to provider-specific
parameters (Anthropic's `thinking.budget_tokens`, OpenAI's `reasoning_effort`, etc.).
litellm translates for each provider.

Values: `"minimal"`, `"low"`, `"medium"`, `"high"`, `"xhigh"` (or `None` for default).


```python
# Compare reasoning across all target models and effort levels
QUESTION = "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost? Think carefully."

for model in ALL_MODELS:
    for effort in [None, "high"]:
        context = AgentContext(
            system_prompt="Be concise. Give the answer and brief reasoning.",
            messages=[],
            tools=None,
        )
        config = AgentConfig(
            model=model,
            convert_to_llm=make_default_convert(model),
            reasoning_effort=effort,
        )
        stream = agent_loop(
            [{"role": "user", "content": QUESTION}],
            context,
            config,
        )
        result = await stream.result()
        assistant = [m for m in result if m.get("role") == "assistant"][-1]
        thinking = assistant.get("reasoning_content")
        print(f"{model} | reasoning_effort={effort!r}")
        print(f"  answer: {assistant.get('content')[:120]}")
        print(f"  thinking: {thinking[:100] + '...' if thinking else '(none)'}")
        print(f"  tokens: {assistant.get('usage', {}).get('total_tokens')}")
        print()
```

    anthropic/claude-sonnet-4-6 | reasoning_effort=None
      answer: ## The Ball Costs **$0.05** (5 cents)
    
    ### Reasoning:
    
    Let the ball cost **x**.
    Then the bat costs **x + $1.00**.
    
    Toget
      thinking: (none)
      tokens: 243
    
    anthropic/claude-sonnet-4-6 | reasoning_effort='high'
      answer: ## The ball costs **$0.05** (5 cents).
    
    **Reasoning:**
    
    Let the ball = x
    Then the bat = x + $1.00
    
    Together: x + (x + $1
      thinking: The ball costs $0.05.
    
    If the ball costs x, then the bat costs x + 1.00.
    x + (x + 1.00) = 1.10
    2x = ...
      tokens: 297
    
    anthropic/claude-opus-4-6 | reasoning_effort=None
      answer: # The Ball Costs **$0.05**
    
    **Reasoning:**
    
    Let the ball's cost = *x*
    
    The bat costs $1.00 more than the ball, so the ba
      thinking: (none)
      tokens: 230
    
    anthropic/claude-opus-4-6 | reasoning_effort='high'
      answer: The ball costs **$0.05** (5 cents).
    
    **Reasoning:** Let the ball = *x*. The bat = *x* + $1.00. Together: *x* + (*x* + $1
      thinking: Let the ball cost x dollars. The bat costs x + $1.00. Together: x + (x + $1.00) = $1.10, so 2x = $0....
      tokens: 235
    
    gemini/gemini-3-flash-preview | reasoning_effort=None
      answer: The ball costs **$0.05** (5 cents).
    
    **Reasoning:**
    If the ball costs $x$, the bat costs $x + $1.00. 
    Combined: $x + (x 
      thinking: (none)
      tokens: 470
    
    gemini/gemini-3-flash-preview | reasoning_effort='high'
      answer: The ball costs **$0.05** (5 cents).
    
    **Reasoning:**
    If the ball costs $0.05 and the bat costs $1.05 ($1.00 more than the
      thinking: **Calculating the Solution**
    
    I've set up the variables and am moving on to the equation! I've defin...
      tokens: 427
    
    gpt-5.2 | reasoning_effort=None
      answer: Let the ball cost \(x\). Then the bat costs \(x + 1.00\).
    
    Total: \(x + (x + 1.00) = 1.10 \Rightarrow 2x = 0.10 \Rightar
      thinking: (none)
      tokens: 133
    
    gpt-5.2 | reasoning_effort='high'
      answer: Let the ball cost \(x\). Then the bat costs \(x + \$1.00\).
    
    Total: \(x + (x + 1.00) = 1.10 \Rightarrow 2x = 0.10 \Right
      thinking: (none)
      tokens: 155
    


## 17. `ToolResult.details` — UI-only data

`ToolResult` has two fields: `content` (sent to the LLM) and `details` (UI-only, never sent to LLM).

This split lets tools return rich metadata for the UI (interactive charts, syntax highlighting,
source URLs) without polluting what the LLM sees. The loop carries `details` through events
so the UI can render them, but `convert_to_llm` strips them before the LLM call.


```python
# A "lookup" tool: content has the answer text, details has rich UI metadata
async def lookup_execute(tool_call_id, params, signal=None, on_update=None):
    return ToolResult(
        content=[{"type": "text", "text": "The population of Tokyo is 14 million."}],
        details={
            "source_url": "https://example.com/tokyo",
            "confidence": 0.95,
            "chart_html": "<div>interactive population chart</div>",
        },
    )


lookup_tool = Tool(
    name="lookup",
    description="Look up a fact. Use when asked about factual data.",
    parameters={
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    },
    execute=lookup_execute,
)

# Wrap convert_to_llm to capture what actually gets sent to the LLM
llm_calls = []
base_convert = make_default_convert(MODEL)


def logging_convert(messages):
    converted = base_convert(messages)
    llm_calls.append(converted)
    return converted


context = AgentContext(
    system_prompt="Use the lookup tool when asked about facts. Be concise.",
    messages=[],
    tools=[lookup_tool],
)
config = AgentConfig(model=MODEL, convert_to_llm=logging_convert)

stream = agent_loop(
    [{"role": "user", "content": "What is the population of Tokyo?"}],
    context,
    config,
)

async for event in stream:
    if event["type"] == "message_update":
        continue
    print(event)

# Now show what the LLM actually saw on the second call (after tool result)
print("\n=== What the LLM received (2nd call, after tool executed) ===")
for msg in llm_calls[1]:
    print(f"  [{msg['role']}] keys={list(msg.keys())}")
    if msg["role"] == "tool":
        print(f"         content={msg['content']!r:.60s}")
        print(f"         has 'details': {'details' in msg}")  # should be False
```

    {'type': 'agent_start'}
    {'type': 'turn_start'}
    {'type': 'message_start', 'message': {'role': 'user', 'content': 'What is the population of Tokyo?'}}
    {'type': 'message_end', 'message': {'role': 'user', 'content': 'What is the population of Tokyo?'}}
    {'type': 'message_start', 'message': {'role': 'assistant', 'content': None, 'tool_calls': None}}
    {'type': 'message_end', 'message': {'role': 'assistant', 'content': None, 'tool_calls': [{'id': 'toolu_01LaDPuVhaJNSb3RQnCj5cxF', 'type': 'function', 'function': {'name': 'lookup', 'arguments': '{"query": "population of Tokyo"}'}, 'provider_specific_fields': None}], 'thinking_blocks': None, 'reasoning_content': None, 'provider_specific_fields': None, 'model': 'anthropic/claude-sonnet-4-6', 'usage': {'prompt_tokens': 583, 'completion_tokens': 54, 'total_tokens': 637, 'cache_read_tokens': 0, 'cache_creation_tokens': 0}, 'stop_reason': 'tool_calls', 'timestamp': 1773398290621}}
    {'type': 'tool_execution_start', 'tool_call_id': 'toolu_01LaDPuVhaJNSb3RQnCj5cxF', 'tool_name': 'lookup', 'args': {'query': 'population of Tokyo'}}
    {'type': 'tool_execution_end', 'tool_call_id': 'toolu_01LaDPuVhaJNSb3RQnCj5cxF', 'tool_name': 'lookup', 'result': {'content': [{'type': 'text', 'text': 'The population of Tokyo is 14 million.'}], 'details': {'source_url': 'https://example.com/tokyo', 'confidence': 0.95, 'chart_html': '<div>interactive population chart</div>'}}, 'is_error': False}
    {'type': 'message_start', 'message': {'role': 'tool', 'tool_call_id': 'toolu_01LaDPuVhaJNSb3RQnCj5cxF', 'name': 'lookup', 'content': [{'type': 'text', 'text': 'The population of Tokyo is 14 million.'}], 'details': {'source_url': 'https://example.com/tokyo', 'confidence': 0.95, 'chart_html': '<div>interactive population chart</div>'}, 'is_error': False, 'timestamp': 1773398290621}}
    {'type': 'message_end', 'message': {'role': 'tool', 'tool_call_id': 'toolu_01LaDPuVhaJNSb3RQnCj5cxF', 'name': 'lookup', 'content': [{'type': 'text', 'text': 'The population of Tokyo is 14 million.'}], 'details': {'source_url': 'https://example.com/tokyo', 'confidence': 0.95, 'chart_html': '<div>interactive population chart</div>'}, 'is_error': False, 'timestamp': 1773398290621}}
    {'type': 'turn_end', 'message': {'role': 'assistant', 'content': None, 'tool_calls': [{'id': 'toolu_01LaDPuVhaJNSb3RQnCj5cxF', 'type': 'function', 'function': {'name': 'lookup', 'arguments': '{"query": "population of Tokyo"}'}, 'provider_specific_fields': None}], 'thinking_blocks': None, 'reasoning_content': None, 'provider_specific_fields': None, 'model': 'anthropic/claude-sonnet-4-6', 'usage': {'prompt_tokens': 583, 'completion_tokens': 54, 'total_tokens': 637, 'cache_read_tokens': 0, 'cache_creation_tokens': 0}, 'stop_reason': 'tool_calls', 'timestamp': 1773398290621}, 'tool_results': [{'role': 'tool', 'tool_call_id': 'toolu_01LaDPuVhaJNSb3RQnCj5cxF', 'name': 'lookup', 'content': [{'type': 'text', 'text': 'The population of Tokyo is 14 million.'}], 'details': {'source_url': 'https://example.com/tokyo', 'confidence': 0.95, 'chart_html': '<div>interactive population chart</div>'}, 'is_error': False, 'timestamp': 1773398290621}]}
    {'type': 'turn_start'}
    {'type': 'message_start', 'message': {'role': 'assistant', 'content': None, 'tool_calls': None}}
    {'type': 'message_end', 'message': {'role': 'assistant', 'content': 'The population of Tokyo is approximately **14 million** people. Note that this refers to the city proper. When including the greater Tokyo metropolitan area, the population is significantly larger, making it one of the most populous metropolitan areas in the world.', 'tool_calls': None, 'thinking_blocks': None, 'reasoning_content': None, 'provider_specific_fields': None, 'model': 'anthropic/claude-sonnet-4-6', 'usage': {'prompt_tokens': 658, 'completion_tokens': 52, 'total_tokens': 710, 'cache_read_tokens': 0, 'cache_creation_tokens': 0}, 'stop_reason': 'stop', 'timestamp': 1773398292261}}
    {'type': 'turn_end', 'message': {'role': 'assistant', 'content': 'The population of Tokyo is approximately **14 million** people. Note that this refers to the city proper. When including the greater Tokyo metropolitan area, the population is significantly larger, making it one of the most populous metropolitan areas in the world.', 'tool_calls': None, 'thinking_blocks': None, 'reasoning_content': None, 'provider_specific_fields': None, 'model': 'anthropic/claude-sonnet-4-6', 'usage': {'prompt_tokens': 658, 'completion_tokens': 52, 'total_tokens': 710, 'cache_read_tokens': 0, 'cache_creation_tokens': 0}, 'stop_reason': 'stop', 'timestamp': 1773398292261}, 'tool_results': []}
    {'type': 'agent_end', 'messages': [{'role': 'user', 'content': 'What is the population of Tokyo?'}, {'role': 'assistant', 'content': None, 'tool_calls': [{'id': 'toolu_01LaDPuVhaJNSb3RQnCj5cxF', 'type': 'function', 'function': {'name': 'lookup', 'arguments': '{"query": "population of Tokyo"}'}, 'provider_specific_fields': None}], 'thinking_blocks': None, 'reasoning_content': None, 'provider_specific_fields': None, 'model': 'anthropic/claude-sonnet-4-6', 'usage': {'prompt_tokens': 583, 'completion_tokens': 54, 'total_tokens': 637, 'cache_read_tokens': 0, 'cache_creation_tokens': 0}, 'stop_reason': 'tool_calls', 'timestamp': 1773398290621}, {'role': 'tool', 'tool_call_id': 'toolu_01LaDPuVhaJNSb3RQnCj5cxF', 'name': 'lookup', 'content': [{'type': 'text', 'text': 'The population of Tokyo is 14 million.'}], 'details': {'source_url': 'https://example.com/tokyo', 'confidence': 0.95, 'chart_html': '<div>interactive population chart</div>'}, 'is_error': False, 'timestamp': 1773398290621}, {'role': 'assistant', 'content': 'The population of Tokyo is approximately **14 million** people. Note that this refers to the city proper. When including the greater Tokyo metropolitan area, the population is significantly larger, making it one of the most populous metropolitan areas in the world.', 'tool_calls': None, 'thinking_blocks': None, 'reasoning_content': None, 'provider_specific_fields': None, 'model': 'anthropic/claude-sonnet-4-6', 'usage': {'prompt_tokens': 658, 'completion_tokens': 52, 'total_tokens': 710, 'cache_read_tokens': 0, 'cache_creation_tokens': 0}, 'stop_reason': 'stop', 'timestamp': 1773398292261}]}
    
    === What the LLM received (2nd call, after tool executed) ===
      [user] keys=['role', 'content']
      [assistant] keys=['role', 'content', 'tool_calls', 'thinking_blocks', 'reasoning_content', 'provider_specific_fields', 'model']
      [tool] keys=['role', 'tool_call_id', 'name', 'content']
             content=[{'type': 'text', 'text': 'The population of Tokyo is 14 mil
             has 'details': False



```python
await stream.result()
```




    [{'role': 'user', 'content': 'What is the population of Tokyo?'},
     {'role': 'assistant',
      'content': None,
      'tool_calls': [{'id': 'toolu_01LaDPuVhaJNSb3RQnCj5cxF',
        'type': 'function',
        'function': {'name': 'lookup',
         'arguments': '{"query": "population of Tokyo"}'},
        'provider_specific_fields': None}],
      'thinking_blocks': None,
      'reasoning_content': None,
      'provider_specific_fields': None,
      'model': 'anthropic/claude-sonnet-4-6',
      'usage': {'prompt_tokens': 583,
       'completion_tokens': 54,
       'total_tokens': 637,
       'cache_read_tokens': 0,
       'cache_creation_tokens': 0},
      'stop_reason': 'tool_calls',
      'timestamp': 1773398290621},
     {'role': 'tool',
      'tool_call_id': 'toolu_01LaDPuVhaJNSb3RQnCj5cxF',
      'name': 'lookup',
      'content': [{'type': 'text',
        'text': 'The population of Tokyo is 14 million.'}],
      'details': {'source_url': 'https://example.com/tokyo',
       'confidence': 0.95,
       'chart_html': '<div>interactive population chart</div>'},
      'is_error': False,
      'timestamp': 1773398290621},
     {'role': 'assistant',
      'content': 'The population of Tokyo is approximately **14 million** people. Note that this refers to the city proper. When including the greater Tokyo metropolitan area, the population is significantly larger, making it one of the most populous metropolitan areas in the world.',
      'tool_calls': None,
      'thinking_blocks': None,
      'reasoning_content': None,
      'provider_specific_fields': None,
      'model': 'anthropic/claude-sonnet-4-6',
      'usage': {'prompt_tokens': 658,
       'completion_tokens': 52,
       'total_tokens': 710,
       'cache_read_tokens': 0,
       'cache_creation_tokens': 0},
      'stop_reason': 'stop',
      'timestamp': 1773398292261}]



## 18. Multimodal tools + `make_default_convert`

Tools can return images alongside text. The interesting part is what happens at the
provider boundary — because **OpenAI doesn't support images in tool result messages**.

**Anthropic / Gemini:** Tool results accept content block arrays with `image_url` blocks.
The image goes directly in the tool result message. Simple.

**OpenAI (GPT-5.2):** Tool results are **string-only**. If you put an image in a tool result,
OpenAI silently drops it. The workaround: strip images from tool results, re-inject them
as a synthetic **user** message after the tool result. The LLM still sees the image — just
via a different message type.

`make_default_convert` handles this automatically — it detects the model provider and
hoists images for OpenAI, passes them natively for Anthropic/Gemini. No custom converter needed.

We'll use a chart tool that returns text + a bar chart image, then ask the LLM about the
chart in a follow-up turn. This proves the image flows through the conversation and the
LLM can actually see it.


```python
import base64
import io
import random
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Chart tool — returns text + image with one obvious spike
async def chart_execute(tool_call_id, params, signal=None, on_update=None):
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun"]
    values = [random.randint(5, 15) for _ in months]
    # Make one bar 10x bigger so any model can spot it
    spike_idx = random.randint(0, 5)
    values[spike_idx] = 150
    spike_month = months[spike_idx]

    fig, ax = plt.subplots(figsize=(6, 3))
    colors = ["#e74c3c" if i == spike_idx else "#3498db" for i in range(6)]
    ax.bar(months, values, color=colors)
    for i, v in enumerate(values):
        ax.text(i, v + 2, str(v), ha="center", fontsize=10, fontweight="bold")
    ax.set_title(params.get("title", "Monthly Errors"))
    ax.set_ylabel("Count")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=80, bbox_inches="tight")
    plt.close(fig)
    img_b64 = base64.b64encode(buf.getvalue()).decode()

    return ToolResult(
        content=[
            {
                "type": "text",
                "text": f"Chart generated. Months: {months}, Values: {values}",
            },
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{img_b64}"},
            },
        ],
        details={"spike_month": spike_month},
    )


chart_tool = Tool(
    name="generate_chart",
    description="Generate a bar chart of monthly server errors. Returns text + image.",
    parameters={
        "type": "object",
        "properties": {"title": {"type": "string", "description": "Chart title"}},
    },
    execute=chart_execute,
)
```


```python
# Test all models: generate chart → ask which month has the spike
# Use logging_convert to see exactly what each provider receives
MODELS_TO_TEST = [
    "anthropic/claude-sonnet-4-6",
    "anthropic/claude-opus-4-6",
    "gemini/gemini-3-flash-preview",
    "gpt-5.2",
]

for model in MODELS_TO_TEST:
    print(f"\n{'=' * 60}")
    print(f"  {model}")
    print(f"{'=' * 60}")

    llm_calls = []
    base_convert = make_default_convert(model)

    def logging_convert(messages, _base=base_convert, _calls=llm_calls):
        converted = _base(messages)
        _calls.append(converted)
        return converted

    # Turn 1: generate the chart
    context = AgentContext(
        system_prompt="You are a data analyst. Use tools when asked. Be concise.",
        messages=[],
        tools=[chart_tool],
    )
    config = AgentConfig(model=model, convert_to_llm=logging_convert)

    stream = agent_loop(
        [{"role": "user", "content": "Generate a chart of monthly server errors."}],
        context,
        config,
    )

    spike_month = None
    new_messages = []
    async for event in stream:
        if event["type"] == "tool_execution_end":
            spike_month = event["result"]["details"]["spike_month"]
            print(f"  spike month: {spike_month}")
        if (
            event["type"] == "message_end"
            and event["message"].get("role") == "assistant"
        ):
            content = event["message"].get("content")
            if content:
                print(f"  assistant: {content[:100]}...")

    new_messages = await stream.result()
    context.messages.extend(new_messages)

    # Turn 2: ask about the chart (LLM must see the image)
    llm_calls.clear()
    stream2 = agent_loop(
        [
            {
                "role": "user",
                "content": "Which month has the highest error count? Reply with just the month name.",
            }
        ],
        context,
        config,
    )

    answer = ""
    async for event in stream2:
        if (
            event["type"] == "message_end"
            and event["message"].get("role") == "assistant"
        ):
            answer = (event["message"].get("content") or "").strip()

    # Show what the LLM received on turn 2
    print("\n  --- What the LLM received (turn 2) ---")
    if llm_calls:
        for msg in llm_calls[0]:
            role = msg["role"]
            content = msg.get("content", "")
            if role == "tool":
                content_type = type(content).__name__
                if isinstance(content, list):
                    types = [b.get("type") for b in content]
                    print(f"  [{role}] content blocks: {types}")
                else:
                    print(f"  [{role}] content: {content[:60]!r}")
            elif role == "user" and isinstance(content, list):
                types = [b.get("type") for b in content]
                print(
                    f"  [{role}] content blocks: {types}  ← synthetic image injection"
                )
            else:
                c = content if isinstance(content, str) else str(content)
                print(f"  [{role}] {c[:80]!r}")

    # Did the model spot the spike?
    month_map = {
        "jan": "january",
        "feb": "february",
        "mar": "march",
        "apr": "april",
        "may": "may",
        "jun": "june",
    }
    full = month_map.get(spike_month.lower(), spike_month.lower())
    found = spike_month.lower() in answer.lower() or full in answer.lower()
    print(f"\n  answer: {answer!r}")
    print(f"  correct: {found} (expected {spike_month})")
```

    
    ============================================================
      anthropic/claude-sonnet-4-6
    ============================================================


    Unclosed client session
    client_session: <aiohttp.client.ClientSession object at 0x11f2ec890>


      assistant: Sure! Let me generate that chart for you right away....
      spike month: May
      assistant: Here's the **Monthly Server Errors** bar chart! Here are some key takeaways:
    
    - 📅 **Data covers:** J...
    
      --- What the LLM received (turn 2) ---
      [user] 'Generate a chart of monthly server errors.'
      [assistant] 'Sure! Let me generate that chart for you right away.'
      [tool] content blocks: ['text', 'image_url']
      [assistant] "Here's the **Monthly Server Errors** bar chart! Here are some key takeaways:\n\n- "
      [user] 'Which month has the highest error count? Reply with just the month name.'
    
      answer: 'May'
      correct: True (expected May)
    
    ============================================================
      anthropic/claude-opus-4-6
    ============================================================
      spike month: Feb
      assistant: Here's the bar chart of monthly server errors. Key observations:
    
    - **February is a clear outlier** ...
    
      --- What the LLM received (turn 2) ---
      [user] 'Generate a chart of monthly server errors.'
      [assistant] 'None'
      [tool] content blocks: ['text', 'image_url']
      [assistant] "Here's the bar chart of monthly server errors. Key observations:\n\n- **February i"
      [user] 'Which month has the highest error count? Reply with just the month name.'
    
      answer: 'February'
      correct: True (expected Feb)
    
    ============================================================
      gemini/gemini-3-flash-preview
    ============================================================
      spike month: Jan
      assistant: _
    ...
    
      --- What the LLM received (turn 2) ---
      [user] 'Generate a chart of monthly server errors.'
      [assistant] 'None'
      [tool] content blocks: ['text', 'image_url']
      [assistant] '_\n'
      [user] 'Which month has the highest error count? Reply with just the month name.'
    
      answer: 'January'
      correct: True (expected Jan)
    
    ============================================================
      gpt-5.2
    ============================================================
      spike month: Mar
      assistant: Here’s the chart of **monthly server errors** (Jan–Jun):
    
    - **Jan:** 6  
    - **Feb:** 9  
    - **Mar:** 1...
    
      --- What the LLM received (turn 2) ---
      [user] 'Generate a chart of monthly server errors.'
      [assistant] 'None'
      [tool] content: "Chart generated. Months: ['Jan', 'Feb', 'Mar', 'Apr', 'May',"
      [user] content blocks: ['text', 'image_url']  ← synthetic image injection
      [assistant] 'Here’s the chart of **monthly server errors** (Jan–Jun):\n\n- **Jan:** 6  \n- **Feb'
      [user] 'Which month has the highest error count? Reply with just the month name.'
    
      answer: 'Mar'
      correct: True (expected Mar)


# Conclusion

  Everything we've explored here (streaming, tool execution, steering, follow-ups, cancellation, context transforms, etc.)  are the mechanics that run inside every `agent.prompt()` call. The
  loop is the engine and you don't need to interact with it directly like we did here.

  In Part 3, we'll use the Agent class on top of this. That's the public API. It's where you actually configure models, queue steering messages, subscribe to events, and manage conversation
  state.
