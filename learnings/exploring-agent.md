---
title: Part 3 of Porting pi-mono's agent loop to Python - The Stateful Wrapper Agent
author: Chris Levy
draft: false
date: '2026-03-13'
date-modified: '2026-03-13'
image: agent_wrapper.jpg
description:  "Part 3 of porting pi-mono's agent loop to Python. We explore the stateful Agent wrapper that sits on top of the raw loop, managing conversation history, tool registration, and
  multi-turn interactions." 
tags:
  - python
  - pi-mono
  - agents
  - llm
  - liteagent
---
# Introduction

This is **Part 3** in a series where I port [pi-mono](https://github.com/badlogic/pi-mono)'s TypeScript agent loop to Python as [liteagent](https://github.com/DrChrisLevy/liteagent).

  - [Part 1: Async Event Streams](https://drchrislevy.com/blog/blog_post?fpath=posts%2Fasync_event_streams%2Fasync_event_streams.ipynb) — the streaming foundation
  - [Part 2: The Raw Loop](https://drchrislevy.com/blog/blog_post?fpath=posts%2Fdual_loop%2Fexploring-loop.ipynb) — the core LLM  tool loop

  In Part 2 we built the raw loop, a stateless function that makes a single LLM call, executes tools, and yields events. Now we wrap it with the `Agent` class: a stateful layer that manages
  conversation history, tool registration, and multi-turn interactions. This is where the raw loop becomes something you can actually use.

# Exploring agent.py — The Stateful Wrapper

This notebook is a hands-on walkthrough of `liteagent.Agent` — the stateful class that wraps
the raw loop.


The raw loop (`agent_loop`, `agent_loop_continue`) that we saw in Part 2 is stateless. You pass in context, it returns
an EventStream, you iterate events yourself. 

The `Agent` class wraps the loop and manages:
- **Message history** — accumulates across turns, no manual context threading
- **Event subscription** — `subscribe(callback)` instead of manual `async for`
- **Steering + follow-up queues** — `steer()` and `follow_up()` with dequeue modes
- **Cancellation** — `abort()` with partial message preservation
- **State tracking** — `is_streaming`, `stream_message`, `pending_tool_calls`, `error`

This is the same two-layer design as pi-mono (`agent-loop.ts` + `agent.ts`).

**What we'll cover:**
1. Setup + imports
2. Simplest prompt — string in, messages out
3. `subscribe()` — the primary consumer API
4. `prompt()` overloads — string, dict, list, images
5. State access — what you can inspect during and after a run
6. Multi-turn — why Agent is stateful
7. Steering — `steer()` mid-run
8. Follow-up — `follow_up()` after idle
9. Queue modes — one-at-a-time vs all
10. `continue_run()` — resume from context
11. `abort()` and partial preservation
12. `wait_for_idle()`
13. `reset()` vs `clear_messages()`
14. Configuration setters — mid-run changes
15. Error handling
16. `_default_convert_to_llm` — what it does
17. Testing across models
18. Real-world patterns
19. Conclusion

---

## 1. Setup

The Agent needs a model string (litellm format) and optionally tools, system prompt,
and a `convert_to_llm` function. Let's import everything and define our helpers.

```
 pip install git+https://github.com/DrChrisLevy/liteagent.git
```

or

```
uv pip install git+https://github.com/DrChrisLevy/liteagent.git
```

You'll need API keys set as environment variables (e.g. ANTHROPIC_API_KEY, OPENAI_API_KEY). In a notebook you can load them from a .env file:


```python
from dotenv import load_dotenv
load_dotenv()
```




    True




```python
from liteagent import Agent, Tool, ToolResult
from liteagent.convert import make_default_convert

# Default model for all examples
MODEL = "anthropic/claude-sonnet-4-6"

# The Agent has a built-in default converter (make_default_convert) that:
# - Strips liteagent metadata (usage, timestamp, stop_reason, etc.)
# - Passes everything else through (thinking_blocks, reasoning_content,
#   provider_specific_fields — new litellm fields survive automatically)
# - For OpenAI models: hoists images from tool results into user messages
#   (OpenAI ignores image blocks in tool result content)
#
# Most examples below use the default — no convert_to_llm needed.
# We'll explore the converter in detail in section 16.


# Simple echo tool — reused across examples
async def echo_execute(tool_call_id, params, signal=None, on_update=None):
    return ToolResult(content=[{"type": "text", "text": params["message"]}])


echo_tool = Tool(
    name="echo",
    description="Echo back a message exactly",
    parameters={
        "type": "object",
        "properties": {
            "message": {"type": "string", "description": "The message to echo"}
        },
        "required": ["message"],
    },
    execute=echo_execute,
)

print("Setup complete.")
```

    Setup complete.


## 2. Simplest prompt — string in, messages out

The absolute minimum: create an Agent, call `prompt("...")`, check `agent.messages`.

Unlike the raw loop (where you build `AgentContext` + `AgentConfig`, call `agent_loop`,
and iterate the EventStream yourself), the Agent does all of that internally.
`prompt()` blocks until the loop completes.


```python
agent = Agent(
    model=MODEL,
    system_prompt="Be concise. One sentence max.",
)

await agent.prompt("What is 2 + 2?")
```


```python
agent.messages
```




    [{'role': 'user', 'content': 'What is 2 + 2?', 'timestamp': 1773456121257},
     {'role': 'assistant',
      'content': '4',
      'tool_calls': None,
      'thinking_blocks': None,
      'reasoning_content': None,
      'provider_specific_fields': None,
      'model': 'anthropic/claude-sonnet-4-6',
      'usage': {'prompt_tokens': 25,
       'completion_tokens': 5,
       'total_tokens': 30,
       'cache_read_tokens': 0,
       'cache_creation_tokens': 0},
      'stop_reason': 'stop',
      'timestamp': 1773456123195}]




```python
agent.state.messages
```




    [{'role': 'user', 'content': 'What is 2 + 2?', 'timestamp': 1773456121257},
     {'role': 'assistant',
      'content': '4',
      'tool_calls': None,
      'thinking_blocks': None,
      'reasoning_content': None,
      'provider_specific_fields': None,
      'model': 'anthropic/claude-sonnet-4-6',
      'usage': {'prompt_tokens': 25,
       'completion_tokens': 5,
       'total_tokens': 30,
       'cache_read_tokens': 0,
       'cache_creation_tokens': 0},
      'stop_reason': 'stop',
      'timestamp': 1773456123195}]




```python
from pprint import pprint
pprint(agent.state)
```

    AgentState(system_prompt='Be concise. One sentence max.',
               model='anthropic/claude-sonnet-4-6',
               thinking_level='off',
               tools=[],
               messages=[{'content': 'What is 2 + 2?',
                          'role': 'user',
                          'timestamp': 1773456121257},
                         {'content': '4',
                          'model': 'anthropic/claude-sonnet-4-6',
                          'provider_specific_fields': None,
                          'reasoning_content': None,
                          'role': 'assistant',
                          'stop_reason': 'stop',
                          'thinking_blocks': None,
                          'timestamp': 1773456123195,
                          'tool_calls': None,
                          'usage': {'cache_creation_tokens': 0,
                                    'cache_read_tokens': 0,
                                    'completion_tokens': 5,
                                    'prompt_tokens': 25,
                                    'total_tokens': 30}}],
               is_streaming=False,
               stream_message=None,
               pending_tool_calls=set(),
               error=None)



```python
agent.state.messages
```




    [{'role': 'user', 'content': 'What is 2 + 2?', 'timestamp': 1773456121257},
     {'role': 'assistant',
      'content': '4',
      'tool_calls': None,
      'thinking_blocks': None,
      'reasoning_content': None,
      'provider_specific_fields': None,
      'model': 'anthropic/claude-sonnet-4-6',
      'usage': {'prompt_tokens': 25,
       'completion_tokens': 5,
       'total_tokens': 30,
       'cache_read_tokens': 0,
       'cache_creation_tokens': 0},
      'stop_reason': 'stop',
      'timestamp': 1773456123195}]




```python
# The default converter strips liteagent metadata, keeps LLM-compatible fields
convert = make_default_convert(MODEL)
convert(agent.state.messages)
```




    [{'role': 'user', 'content': 'What is 2 + 2?'},
     {'role': 'assistant',
      'content': '4',
      'tool_calls': None,
      'thinking_blocks': None,
      'reasoning_content': None,
      'provider_specific_fields': None,
      'model': 'anthropic/claude-sonnet-4-6'}]




```python
# User message — our input, wrapped in a dict by prompt()
agent.messages[0]
```




    {'role': 'user', 'content': 'What is 2 + 2?', 'timestamp': 1773456121257}




```python
# Assistant message — enriched with usage, stop_reason, timestamp, model, etc.
agent.messages[1]
```




    {'role': 'assistant',
     'content': '4',
     'tool_calls': None,
     'thinking_blocks': None,
     'reasoning_content': None,
     'provider_specific_fields': None,
     'model': 'anthropic/claude-sonnet-4-6',
     'usage': {'prompt_tokens': 25,
      'completion_tokens': 5,
      'total_tokens': 30,
      'cache_read_tokens': 0,
      'cache_creation_tokens': 0},
     'stop_reason': 'stop',
     'timestamp': 1773456123195}




```python
# This is why we usd @property --> Read-only access. With @property, it prevents replacing the state object itself.
agent.state = "BREAK THE STATE"
```


    ---------------------------------------------------------------------------

    AttributeError                            Traceback (most recent call last)

    Cell In[51], line 2
          1 # This is why we usd @property --> Read-only access. With @property, it prevents replacing the state object itself.
    ----> 2 agent.state = "BREAK THE STATE"


    AttributeError: property 'state' of 'Agent' object has no setter


The assistant message has all the extras the loop adds:
- `usage` — token counts from litellm
- `stop_reason` — "stop" (normal), "tool_calls", "error", "aborted"
- `timestamp` — Unix ms
- `thinking_blocks` / `reasoning_content` — None unless thinking is enabled
- `provider_specific_fields` — opaque bag from litellm

These extras are why `convert_to_llm` exists — they must be stripped before
sending messages back to the LLM.

## 3. subscribe()— the primary consumer API

In the [loop notebook](https://drchrislevy.com/blog/blog_post?fpath=posts%2Fdual_loop%2Fexploring-loop.ipynb), we used `async for event in stream` to consume events.
The `Agent` doesn't expose the stream. Instead, you subscribe a callback:

```python
unsub = agent.subscribe(my_callback)  # returns unsubscribe function
```

The callback fires synchronously during `await agent.prompt()` using the same thread so there are no concurrency issues. 
This is how pi's agent works too.

**Why callbacks instead of async iteration?** 

The Agent is the sole reader of
the loop's EventStream (internal detail). External consumers get events via
subscribe. This lets multiple consumers see the same events (unlike a queue
where each item is consumed once).


```python
agent = Agent(
    model=MODEL,
    system_prompt="Be concise.",
)

# Collect all events
events = []
unsub = agent.subscribe(lambda e: events.append(e))

await agent.prompt("Say hello!")
```


```python
for e in events:
    print(e)
```

    {'type': 'agent_start'}
    {'type': 'turn_start'}
    {'type': 'message_start', 'message': {'role': 'user', 'content': 'Say hello!', 'timestamp': 1773456211781}}
    {'type': 'message_end', 'message': {'role': 'user', 'content': 'Say hello!', 'timestamp': 1773456211781}}
    {'type': 'message_start', 'message': {'role': 'assistant', 'content': None, 'tool_calls': None}}
    {'type': 'message_update', 'message': {'role': 'assistant', 'content': 'Hello! ', 'tool_calls': None}, 'delta': {'content': 'Hello! '}, 'delta_type': 'text_delta'}
    {'type': 'message_update', 'message': {'role': 'assistant', 'content': 'Hello! 👋 How are', 'tool_calls': None}, 'delta': {'content': '👋 How are'}, 'delta_type': 'text_delta'}
    {'type': 'message_update', 'message': {'role': 'assistant', 'content': 'Hello! 👋 How are you doing?', 'tool_calls': None}, 'delta': {'content': ' you doing?'}, 'delta_type': 'text_delta'}
    {'type': 'message_update', 'message': {'role': 'assistant', 'content': 'Hello! 👋 How are you doing? Is', 'tool_calls': None}, 'delta': {'content': ' Is'}, 'delta_type': 'text_delta'}
    {'type': 'message_update', 'message': {'role': 'assistant', 'content': 'Hello! 👋 How are you doing? Is there something I can help you with today', 'tool_calls': None}, 'delta': {'content': ' there something I can help you with today'}, 'delta_type': 'text_delta'}
    {'type': 'message_update', 'message': {'role': 'assistant', 'content': 'Hello! 👋 How are you doing? Is there something I can help you with today?', 'tool_calls': None}, 'delta': {'content': '?'}, 'delta_type': 'text_delta'}
    {'type': 'message_end', 'message': {'role': 'assistant', 'content': 'Hello! 👋 How are you doing? Is there something I can help you with today?', 'tool_calls': None, 'thinking_blocks': None, 'reasoning_content': None, 'provider_specific_fields': None, 'model': 'anthropic/claude-sonnet-4-6', 'usage': {'prompt_tokens': 15, 'completion_tokens': 24, 'total_tokens': 39, 'cache_read_tokens': 0, 'cache_creation_tokens': 0}, 'stop_reason': 'stop', 'timestamp': 1773456213430}}
    {'type': 'turn_end', 'message': {'role': 'assistant', 'content': 'Hello! 👋 How are you doing? Is there something I can help you with today?', 'tool_calls': None, 'thinking_blocks': None, 'reasoning_content': None, 'provider_specific_fields': None, 'model': 'anthropic/claude-sonnet-4-6', 'usage': {'prompt_tokens': 15, 'completion_tokens': 24, 'total_tokens': 39, 'cache_read_tokens': 0, 'cache_creation_tokens': 0}, 'stop_reason': 'stop', 'timestamp': 1773456213430}, 'tool_results': []}
    {'type': 'agent_end', 'messages': [{'role': 'user', 'content': 'Say hello!', 'timestamp': 1773456211781}, {'role': 'assistant', 'content': 'Hello! 👋 How are you doing? Is there something I can help you with today?', 'tool_calls': None, 'thinking_blocks': None, 'reasoning_content': None, 'provider_specific_fields': None, 'model': 'anthropic/claude-sonnet-4-6', 'usage': {'prompt_tokens': 15, 'completion_tokens': 24, 'total_tokens': 39, 'cache_read_tokens': 0, 'cache_creation_tokens': 0}, 'stop_reason': 'stop', 'timestamp': 1773456213430}]}


Same events sequence as the raw loop:

Now let's test unsubscribe:


```python
count_before = len(events)
count_before
```




    14




```python
unsub()  # stop receiving events
```


```python
agent._subscribers
```




    []




```python
await agent.prompt("Say goodbye.")

count_after = len(events)
print(f"Events before unsub: {count_before}")
print(f"Events after second prompt: {count_after}")
print(f"Unsubscribe worked: {count_before == count_after}")
```

    Events before unsub: 14
    Events after second prompt: 14
    Unsubscribe worked: True



```python
agent.state.messages
```




    [{'role': 'user', 'content': 'Say hello!', 'timestamp': 1773456211781},
     {'role': 'assistant',
      'content': 'Hello! 👋 How are you doing? Is there something I can help you with today?',
      'tool_calls': None,
      'thinking_blocks': None,
      'reasoning_content': None,
      'provider_specific_fields': None,
      'model': 'anthropic/claude-sonnet-4-6',
      'usage': {'prompt_tokens': 15,
       'completion_tokens': 24,
       'total_tokens': 39,
       'cache_read_tokens': 0,
       'cache_creation_tokens': 0},
      'stop_reason': 'stop',
      'timestamp': 1773456213430},
     {'role': 'user', 'content': 'Say goodbye.', 'timestamp': 1773456241225},
     {'role': 'assistant',
      'content': 'Goodbye! 👋 Take care, and feel free to come back anytime if you need anything! 😊',
      'tool_calls': None,
      'thinking_blocks': None,
      'reasoning_content': None,
      'provider_specific_fields': None,
      'model': 'anthropic/claude-sonnet-4-6',
      'usage': {'prompt_tokens': 45,
       'completion_tokens': 29,
       'total_tokens': 74,
       'cache_read_tokens': 0,
       'cache_creation_tokens': 0},
      'stop_reason': 'stop',
      'timestamp': 1773456242701}]



## 4. prompt() overloads

Like pi's `agent.prompt()`, ours accepts four input shapes:

| Input | What happens |
|-------|-------------|
| `prompt("string")` | Wrapped in `{"role": "user", "content": "string", "timestamp": ...}` |
| `prompt({"role": "user", ...})` | Used as-is |
| `prompt([msg1, msg2])` | Multiple messages injected |
| `prompt("text", images=[...])` | Multimodal: text + images in content array |

Let's see what each produces.


```python
# Overload 1: string
agent = Agent(model=MODEL)
await agent.prompt("Hello from a string")
agent.messages[0]
```




    {'role': 'user', 'content': 'Hello from a string', 'timestamp': 1773456245841}




```python
# Overload 2: dict (used as-is)
agent = Agent(model=MODEL)
await agent.prompt(
    {"role": "user", "content": "Hello from a dict", "custom_field": "preserved"}
)
agent.messages[0]
```




    {'role': 'user', 'content': 'Hello from a dict', 'custom_field': 'preserved'}




```python
convert = make_default_convert(MODEL)
convert(agent.messages)
```




    [{'role': 'user', 'content': 'Hello from a dict', 'custom_field': 'preserved'},
     {'role': 'assistant',
      'content': 'Hello! It looks like your message came through as plain text rather than a dictionary. 😄\n\nDid you mean to send something like:\n\n```python\n{"message": "Hello"}\n```\n\nOr were you just saying hello in a fun way? Either way, **hello back to you!** 👋\n\nHow can I help you today?',
      'tool_calls': None,
      'thinking_blocks': None,
      'reasoning_content': None,
      'provider_specific_fields': None,
      'model': 'anthropic/claude-sonnet-4-6'}]




```python
# Overload 3: list of messages
agent = Agent(model=MODEL, system_prompt="Be concise.")
await agent.prompt(
    [
        {"role": "user", "content": "My name is Alice."},
        {"role": "user", "content": "What is my name?"},
    ]
)
```


```python
agent.messages
```




    [{'role': 'user', 'content': 'My name is Alice.'},
     {'role': 'user', 'content': 'What is my name?'},
     {'role': 'assistant',
      'content': 'Your name is **Alice**! You just told me. 😊',
      'tool_calls': None,
      'thinking_blocks': None,
      'reasoning_content': None,
      'provider_specific_fields': None,
      'model': 'anthropic/claude-sonnet-4-6',
      'usage': {'prompt_tokens': 22,
       'completion_tokens': 18,
       'total_tokens': 40,
       'cache_read_tokens': 0,
       'cache_creation_tokens': 0},
      'stop_reason': 'stop',
      'timestamp': 1773456255269}]




```python
# Overload 4: string + images (multimodal)
# Send a real image and ask the LLM about it

image_block = {
    "type": "image_url",
    "image_url": {
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/7/7a/LeBron_James_%2851959977144%29_%28cropped2%29.jpg/250px-LeBron_James_%2851959977144%29_%28cropped2%29.jpg"
    },
}

agent = Agent(
    model=MODEL,
    system_prompt="Be concise. One sentence max.",
)
await agent.prompt("What is in this image?", images=[image_block])

print(f"Response: {agent.messages[-1].get('content')}")
```

    Response: A basketball player wearing a **Los Angeles Lakers #6 jersey** is holding a basketball during an NBA game.



```python
agent.messages
```




    [{'role': 'user',
      'content': [{'type': 'text', 'text': 'What is in this image?'},
       {'type': 'image_url',
        'image_url': {'url': 'https://upload.wikimedia.org/wikipedia/commons/thumb/7/7a/LeBron_James_%2851959977144%29_%28cropped2%29.jpg/250px-LeBron_James_%2851959977144%29_%28cropped2%29.jpg'}}],
      'timestamp': 1773456257246},
     {'role': 'assistant',
      'content': 'A basketball player wearing a **Los Angeles Lakers #6 jersey** is holding a basketball during an NBA game.',
      'tool_calls': None,
      'thinking_blocks': None,
      'reasoning_content': None,
      'provider_specific_fields': None,
      'model': 'anthropic/claude-sonnet-4-6',
      'usage': {'prompt_tokens': 124,
       'completion_tokens': 26,
       'total_tokens': 150,
       'cache_read_tokens': 0,
       'cache_creation_tokens': 0},
      'stop_reason': 'stop',
      'timestamp': 1773456259901}]



## 5. State access

The Agent tracks state in an `AgentState` dataclass. You can inspect it at any time:

```python
agent.state.is_streaming       # True while loop is running
agent.state.stream_message     # current partial message being streamed (or None)
agent.state.pending_tool_calls # set of tool call IDs currently executing
agent.state.error              # last error message (or None)
agent.state.model              # current model string
agent.state.system_prompt      # current system prompt
agent.state.tools              # current tool list
agent.state.thinking_level     # "off", "minimal", "low", "medium", "high", "xhigh"
agent.messages                 # shorthand for agent.state.messages
```

Let's watch state change *during* a run using a subscriber:


```python
agent = Agent(
    model=MODEL,
    system_prompt="Be concise.",
    tools=[echo_tool],
)

# Track state transitions
state_log = []

print(
    f"{'Event':<26} {'Streaming':<10} {'StreamMsg':<10} {'PendTools':<10} {'MsgCount':<10}"
)


def track_state(event):
    t = event["type"]
    entry = {
        "event": t,
        "is_streaming": agent.state.is_streaming,
        "stream_msg": agent.state.stream_message is not None,
        "pending_tools": len(agent.state.pending_tool_calls),
        "msg_count": len(agent.messages),
    }
    state_log.append(entry)
    print("-" * 66)
    print(
        f"{entry['event']:<26} {str(entry['is_streaming']):<10} {str(entry['stream_msg']):<10} {entry['pending_tools']:<10} {entry['msg_count']:<10}"
    )


agent.subscribe(track_state)
await agent.prompt("Echo 'hello world'")
```

    Event                      Streaming  StreamMsg  PendTools  MsgCount  
    ------------------------------------------------------------------
    agent_start                True       False      0          0         
    ------------------------------------------------------------------
    turn_start                 True       False      0          0         
    ------------------------------------------------------------------
    message_start              True       False      0          0         
    ------------------------------------------------------------------
    message_end                True       False      0          1         
    ------------------------------------------------------------------
    message_start              True       True       0          1         
    ------------------------------------------------------------------
    message_update             True       True       0          1         
    ------------------------------------------------------------------
    message_update             True       True       0          1         
    ------------------------------------------------------------------
    message_update             True       True       0          1         
    ------------------------------------------------------------------
    message_update             True       True       0          1         
    ------------------------------------------------------------------
    message_update             True       True       0          1         
    ------------------------------------------------------------------
    message_update             True       True       0          1         
    ------------------------------------------------------------------
    message_end                True       False      0          2         
    ------------------------------------------------------------------
    tool_execution_start       True       False      1          2         
    ------------------------------------------------------------------
    tool_execution_end         True       False      0          2         
    ------------------------------------------------------------------
    message_start              True       False      0          2         
    ------------------------------------------------------------------
    message_end                True       False      0          3         
    ------------------------------------------------------------------
    turn_end                   True       False      0          3         
    ------------------------------------------------------------------
    turn_start                 True       False      0          3         
    ------------------------------------------------------------------
    message_start              True       True       0          3         
    ------------------------------------------------------------------
    message_update             True       True       0          3         
    ------------------------------------------------------------------
    message_update             True       True       0          3         
    ------------------------------------------------------------------
    message_update             True       True       0          3         
    ------------------------------------------------------------------
    message_update             True       True       0          3         
    ------------------------------------------------------------------
    message_update             True       True       0          3         
    ------------------------------------------------------------------
    message_update             True       True       0          3         
    ------------------------------------------------------------------
    message_update             True       True       0          3         
    ------------------------------------------------------------------
    message_end                True       False      0          4         
    ------------------------------------------------------------------
    turn_end                   True       False      0          4         
    ------------------------------------------------------------------
    agent_end                  False      False      0          4         


Notice how:
- `is_streaming` is True throughout the run
- `stream_message` appears on `message_start` (assistant only), disappears on `message_end`
- `pending_tool_calls` increments on `tool_execution_start`, decrements on `tool_execution_end`
- `msg_count` grows on each `message_end` — messages are appended incrementally, not batched


## 6. Multi-turn — why Agent is stateful

This is the Agent's main value: messages persist across `prompt()` calls.
With the raw loop, you'd need to manually thread context between calls.
The Agent does it automatically.


```python
agent = Agent(
    model=MODEL,
    system_prompt="Be concise. Remember everything the user says.",
)

# Turn 1: tell the agent something
await agent.prompt("My favorite color is blue.")
agent.messages
```




    [{'role': 'user',
      'content': 'My favorite color is blue.',
      'timestamp': 1773456267161},
     {'role': 'assistant',
      'content': "Got it — your favorite color is blue! I'll remember that. Is there something I can help you with?",
      'tool_calls': None,
      'thinking_blocks': None,
      'reasoning_content': None,
      'provider_specific_fields': None,
      'model': 'anthropic/claude-sonnet-4-6',
      'usage': {'prompt_tokens': 24,
       'completion_tokens': 26,
       'total_tokens': 50,
       'cache_read_tokens': 0,
       'cache_creation_tokens': 0},
      'stop_reason': 'stop',
      'timestamp': 1773456268874}]




```python
# Turn 2: ask about it — the agent should remember
await agent.prompt("What is my favorite color?")
agent.messages
```




    [{'role': 'user',
      'content': 'My favorite color is blue.',
      'timestamp': 1773456267161},
     {'role': 'assistant',
      'content': "Got it — your favorite color is blue! I'll remember that. Is there something I can help you with?",
      'tool_calls': None,
      'thinking_blocks': None,
      'reasoning_content': None,
      'provider_specific_fields': None,
      'model': 'anthropic/claude-sonnet-4-6',
      'usage': {'prompt_tokens': 24,
       'completion_tokens': 26,
       'total_tokens': 50,
       'cache_read_tokens': 0,
       'cache_creation_tokens': 0},
      'stop_reason': 'stop',
      'timestamp': 1773456268874},
     {'role': 'user',
      'content': 'What is my favorite color?',
      'timestamp': 1773456270373},
     {'role': 'assistant',
      'content': 'Your favorite color is blue!',
      'tool_calls': None,
      'thinking_blocks': None,
      'reasoning_content': None,
      'provider_specific_fields': None,
      'model': 'anthropic/claude-sonnet-4-6',
      'usage': {'prompt_tokens': 59,
       'completion_tokens': 9,
       'total_tokens': 68,
       'cache_read_tokens': 0,
       'cache_creation_tokens': 0},
      'stop_reason': 'stop',
      'timestamp': 1773456272162}]



## 7. Steering — steer() mid-run

`steer()` queues a message that gets injected **during** a run:
- After each tool execution, the loop checks the steering queue
- If there's a message, remaining tools are **skipped** and the steering message
  is injected before the next LLM call
- This is "stop what you're doing, do this instead"

The loop also checks for steering at the start of each run (before the first LLM call).
So if you call `steer()` before `prompt()`, the steering message gets picked up immediately.

Let's demonstrate both: pre-queued steering, and mid-tool steering.


```python
# Pre-queued steering: steer() before prompt()
agent = Agent(
    model=MODEL,
    system_prompt="Be concise.",
)

agent.steer("Actually, tell me a joke instead.")
await agent.prompt("What is the capital of France?")

agent.messages
```




    [{'role': 'user',
      'content': 'What is the capital of France?',
      'timestamp': 1773456273693},
     {'role': 'user',
      'content': 'Actually, tell me a joke instead.',
      'timestamp': 1773456273693},
     {'role': 'assistant',
      'content': "Sure! Here's one:\n\nWhy don't scientists trust atoms?\n\n**Because they make up everything!** 😄",
      'tool_calls': None,
      'thinking_blocks': None,
      'reasoning_content': None,
      'provider_specific_fields': None,
      'model': 'anthropic/claude-sonnet-4-6',
      'usage': {'prompt_tokens': 27,
       'completion_tokens': 29,
       'total_tokens': 56,
       'cache_read_tokens': 0,
       'cache_creation_tokens': 0},
      'stop_reason': 'stop',
      'timestamp': 1773456275370}]




```python
# Mid-tool steering: steer() during tool execution
# When tool_a executes, it queues a steering message.
# tool_b should be SKIPPED.

call_log = []
steering_agent = None


async def tool_a_exec(tool_call_id, params, signal=None, on_update=None):
    call_log.append("a")
    steering_agent.steer("Stop! Do something else.")  # interrupt!
    return ToolResult(content=[{"type": "text", "text": "tool_a done"}])


async def tool_b_exec(tool_call_id, params, signal=None, on_update=None):
    call_log.append("b")
    return ToolResult(content=[{"type": "text", "text": "tool_b done"}])


tool_a = Tool(
    name="tool_a",
    description="Tool A",
    parameters={"type": "object", "properties": {}},
    execute=tool_a_exec,
)
tool_b = Tool(
    name="tool_b",
    description="Tool B",
    parameters={"type": "object", "properties": {}},
    execute=tool_b_exec,
)

steering_agent = Agent(
    model=MODEL,
    system_prompt="When asked, call both tool_a and tool_b in a single response. Be concise.",
    tools=[tool_a, tool_b],
)

await steering_agent.prompt("Call both tool_a and tool_b now.")

steering_agent.messages
```




    [{'role': 'user',
      'content': 'Call both tool_a and tool_b now.',
      'timestamp': 1773456275380},
     {'role': 'assistant',
      'content': 'Sure! Calling both tools simultaneously!',
      'tool_calls': [{'id': 'toolu_01ECJQpPnmsF3sgbYCG4uKf8',
        'type': 'function',
        'function': {'name': 'tool_a', 'arguments': '{}'},
        'provider_specific_fields': None},
       {'id': 'toolu_01D4BVmHMZCMfS1B7wNqpHAn',
        'type': 'function',
        'function': {'name': 'tool_b', 'arguments': '{}'},
        'provider_specific_fields': None}],
      'thinking_blocks': None,
      'reasoning_content': None,
      'provider_specific_fields': None,
      'model': 'anthropic/claude-sonnet-4-6',
      'usage': {'prompt_tokens': 609,
       'completion_tokens': 64,
       'total_tokens': 673,
       'cache_read_tokens': 0,
       'cache_creation_tokens': 0},
      'stop_reason': 'tool_calls',
      'timestamp': 1773456277342},
     {'role': 'tool',
      'tool_call_id': 'toolu_01ECJQpPnmsF3sgbYCG4uKf8',
      'name': 'tool_a',
      'content': [{'type': 'text', 'text': 'tool_a done'}],
      'details': {},
      'is_error': False,
      'timestamp': 1773456277342},
     {'role': 'tool',
      'tool_call_id': 'toolu_01D4BVmHMZCMfS1B7wNqpHAn',
      'name': 'tool_b',
      'content': [{'type': 'text', 'text': 'Skipped due to queued user message.'}],
      'details': {},
      'is_error': True,
      'timestamp': 1773456277342},
     {'role': 'user',
      'content': 'Stop! Do something else.',
      'timestamp': 1773456277342},
     {'role': 'assistant',
      'content': "Sure! I'll stop calling the tools. What would you like me to do instead? Just let me know how I can help!",
      'tool_calls': None,
      'thinking_blocks': None,
      'reasoning_content': None,
      'provider_specific_fields': None,
      'model': 'anthropic/claude-sonnet-4-6',
      'usage': {'prompt_tokens': 760,
       'completion_tokens': 30,
       'total_tokens': 790,
       'cache_read_tokens': 0,
       'cache_creation_tokens': 0},
      'stop_reason': 'stop',
      'timestamp': 1773456279641}]



  1. Assistant calls both tool_a and tool_b
  2. tool_a executes (and queues steering inside its execute function)
  3. tool_b gets skipped — is_error: True, "Skipped due to queued user message."
  4. Steering message injected: "Stop! Do something else."
  5. Assistant responds to the steering instead of continuing

  The key proof: tool_b never ran ('b' not in call_log), but it still has a tool result in the conversation. 
  The LLM needs every tool call to have a result, even skipped ones.

## 8. Follow-up — follow_up() after idle

`follow_up()` is the *outer loop* mechanism. Unlike steering (which interrupts),
follow-ups wait until the agent finishes everything (no more tool calls, no steering).
Then the follow-up message is injected and the agent continues.

- Steering = "stop what you're doing" (immediate)
- Follow-up = "when you're done, also do this" (deferred)


```python
agent = Agent(
    model=MODEL,
    system_prompt="Be concise. One sentence.",
)

# Queue a follow-up BEFORE the first prompt
agent.follow_up("Now tell me a fun fact about cats.")

await agent.prompt("What is 2 + 2?")

agent.messages
```




    [{'role': 'user', 'content': 'What is 2 + 2?', 'timestamp': 1773456281209},
     {'role': 'assistant',
      'content': '4.',
      'tool_calls': None,
      'thinking_blocks': None,
      'reasoning_content': None,
      'provider_specific_fields': None,
      'model': 'anthropic/claude-sonnet-4-6',
      'usage': {'prompt_tokens': 24,
       'completion_tokens': 6,
       'total_tokens': 30,
       'cache_read_tokens': 0,
       'cache_creation_tokens': 0},
      'stop_reason': 'stop',
      'timestamp': 1773456282023},
     {'role': 'user',
      'content': 'Now tell me a fun fact about cats.',
      'timestamp': 1773456281209},
     {'role': 'assistant',
      'content': 'Cats spend about 70% of their lives sleeping.',
      'tool_calls': None,
      'thinking_blocks': None,
      'reasoning_content': None,
      'provider_specific_fields': None,
      'model': 'anthropic/claude-sonnet-4-6',
      'usage': {'prompt_tokens': 42,
       'completion_tokens': 15,
       'total_tokens': 57,
       'cache_read_tokens': 0,
       'cache_creation_tokens': 0},
      'stop_reason': 'stop',
      'timestamp': 1773456283555}]



## 9. Queue modes

Both steering and follow-up have two modes:
- `"one-at-a-time"` (default) — dequeue one message per poll
- `"all"` — dequeue everything at once

This matters when multiple messages are queued. Let's see the difference.


```python
# one-at-a-time (default): queue 3, dequeue returns 1
agent = Agent(model=MODEL)
agent.steer("msg1")
agent.steer("msg2")
agent.steer("msg3")

batch = agent._dequeue_steering()
print(
    f"one-at-a-time: got {len(batch)} message(s), {len(agent._steering_queue)} remaining"
)
print(f"  dequeued: '{batch[0]['content']}'")
```

    one-at-a-time: got 1 message(s), 2 remaining
      dequeued: 'msg1'



```python
# all mode: queue 3, dequeue returns all 3
agent = Agent(model=MODEL, steering_mode="all")
agent.steer("msg1")
agent.steer("msg2")
agent.steer("msg3")

batch = agent._dequeue_steering()
print(f"all mode: got {len(batch)} message(s), {len(agent._steering_queue)} remaining")
for m in batch:
    print(f"  '{m['content']}'")
```

    all mode: got 3 message(s), 0 remaining
      'msg1'
      'msg2'
      'msg3'


## 10. continue_run() — resume from context

`continue_run()` is for when the conversation ended at a tool result or user message
and you want the LLM to continue from there, without sending a new prompt.

Three interesting cases when the last message is an assistant message:
1. Steering queue has messages → use those
2. Follow-up queue has messages → use those
3. Both empty → error (can't continue from assistant without new input)

**When would you actually use this?** In normal chat (`prompt()` → response → `prompt()` again),
you won't. `continue_run()` is for **recovery and resumption** — when something outside the
normal flow modifies the message history. Real-world examples from pi-mono's coding agent:

- **Context compaction** — when the conversation gets too long, old messages are summarized and
  replaced. After compaction the last message might be a tool result the LLM never responded to.
  `continue()` kicks the loop to pick up from there.
- **Error retry** — when an LLM call fails, a "please retry" message is appended and `continue()`
  re-runs the loop without needing a new user prompt.
- **Queued messages after idle** — if `follow_up()` or `steer()` is called after the agent
  finishes, `continue_run()` restarts the loop to process them.


```python
# Case: continue from a tool result (manually built context)
weather_tool = Tool(
    name="weather",
    description="Get the weather in a location",
    parameters={"type": "object", "properties": {"location": {"type": "string"}}},
    execute=lambda *args: ToolResult(content=[{"type": "text", "text": "72°F and sunny in San Francisco"}]),
)
agent = Agent(
    model=MODEL,
    system_prompt="Be concise.",
    tools=[weather_tool],
)

# Simulate: user asked about weather → assistant called tool → we have the result
agent._state.messages = [
    {"role": "user", "content": "What's the weather?"},
    {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "c0",
                "type": "function",
                "function": {"name": "weather", "arguments": "{}"},
            }
        ],
        "stop_reason": "tool_calls",
    },
    {
        "role": "tool",
        "tool_call_id": "c0",
        "content": [{"type": "text", "text": "72°F and sunny in San Francisco"}],
        "is_error": False,
    },
]

await agent.continue_run()

agent.messages
```




    [{'role': 'user', 'content': "What's the weather?"},
     {'role': 'assistant',
      'content': None,
      'tool_calls': [{'id': 'c0',
        'type': 'function',
        'function': {'name': 'weather', 'arguments': '{}'}}],
      'stop_reason': 'tool_calls'},
     {'role': 'tool',
      'tool_call_id': 'c0',
      'content': [{'type': 'text', 'text': '72°F and sunny in San Francisco'}],
      'is_error': False},
     {'role': 'assistant',
      'content': 'Please provide a location so I can get the weather for you. I need a specific city or area to look up the forecast!',
      'tool_calls': None,
      'thinking_blocks': None,
      'reasoning_content': None,
      'provider_specific_fields': None,
      'model': 'anthropic/claude-sonnet-4-6',
      'usage': {'prompt_tokens': 615,
       'completion_tokens': 29,
       'total_tokens': 644,
       'cache_read_tokens': 0,
       'cache_creation_tokens': 0},
      'stop_reason': 'stop',
      'timestamp': 1773456286884}]




```python
# Case: continue from assistant + steering queue
agent = Agent(model=MODEL)
agent._state.messages = [
    {"role": "user", "content": "Hi"},
    {"role": "assistant", "content": "Hello!", "stop_reason": "stop"},
]
agent.steer("Now tell me a joke.")

await agent.continue_run()

agent.messages
```




    [{'role': 'user', 'content': 'Hi'},
     {'role': 'assistant', 'content': 'Hello!', 'stop_reason': 'stop'},
     {'role': 'user',
      'content': 'Now tell me a joke.',
      'timestamp': 1773456286891},
     {'role': 'assistant',
      'content': "Here's one for you:\n\nWhy don't scientists trust atoms?\n\nBecause they make up everything! 😄\n\nWould you like to hear another one?",
      'tool_calls': None,
      'thinking_blocks': None,
      'reasoning_content': None,
      'provider_specific_fields': None,
      'model': 'anthropic/claude-sonnet-4-6',
      'usage': {'prompt_tokens': 22,
       'completion_tokens': 36,
       'total_tokens': 58,
       'cache_read_tokens': 0,
       'cache_creation_tokens': 0},
      'stop_reason': 'stop',
      'timestamp': 1773456288572}]




```python
# Case: continue from assistant + empty queues → error
agent = Agent(model=MODEL)
agent._state.messages = [
    {"role": "user", "content": "Hi"},
    {"role": "assistant", "content": "Hello!", "stop_reason": "stop"},
]

try:
    await agent.continue_run()
except ValueError as e:
    print(f"Got expected error: {e}")
```

    Got expected error: Cannot continue from assistant message without queued messages. Use steer() or follow_up() first.


## 11. abort() and partial preservation

`abort()` sets a signal that stops the loop. But what happens to the partial
assistant message that was being streamed? The Agent handles this edge case
(same as pi's agent.):

1. If the partial has **real content** (non-empty text, reasoning, or named tool call) → **preserve it**
2. If it's just **empty scaffolding** (empty strings, unnamed tool calls) → **discard it**
3. If discarded after abort → raise "Request was aborted" → caught by error handler


```python
# abort() after receiving some text — partial should be preserved
agent = Agent(
    model=MODEL,
    system_prompt="Write a very long essay about the history of computing. At least 5000 words.",
)

chunk_count = 0


def abort_after_chunks(event):
    print("signal is set: " + str(agent._signal.is_set()))
    if agent.state.stream_message:
        print(agent.state.stream_message["content"])
    global chunk_count
    if event["type"] == "message_update" and event.get("delta_type") == "text_delta":
        chunk_count += 1
        if chunk_count >= 5:
            agent.abort()


agent.subscribe(abort_after_chunks)
await agent.prompt("Go ahead.")

print(f"is_streaming: {agent.state.is_streaming}")
print(f"signal cleaned up: {agent._signal is None}")
```

    signal is set: False
    signal is set: False
    signal is set: False
    signal is set: False
    signal is set: False
    None
    signal is set: False
    # The History
    signal is set: False
    # The History of Computing
    signal is set: False
    # The History of Computing: From Ancient
    signal is set: False
    # The History of Computing: From Ancient Calculations to the
    signal is set: False
    # The History of Computing: From Ancient Calculations to the Digital Age
    
    ## Introduction
    
    The
    signal is set: True
    # The History of Computing: From Ancient Calculations to the Digital Age
    
    ## Introduction
    
    The history of computing is one
    signal is set: True
    # The History of Computing: From Ancient Calculations to the Digital Age
    
    ## Introduction
    
    The history of computing is one of the most remarkable
    signal is set: True
    # The History of Computing: From Ancient Calculations to the Digital Age
    
    ## Introduction
    
    The history of computing is one of the most remarkable stories
    signal is set: True
    signal is set: True
    signal is set: True
    is_streaming: False
    signal cleaned up: True



```python
agent.messages
```




    [{'role': 'user', 'content': 'Go ahead.', 'timestamp': 1773456288583},
     {'role': 'assistant',
      'content': '# The History of Computing: From Ancient Calculations to the Digital Age\n\n## Introduction\n\nThe history of computing is one of the most remarkable stories',
      'tool_calls': None,
      'thinking_blocks': None,
      'reasoning_content': None,
      'provider_specific_fields': None,
      'model': 'anthropic/claude-sonnet-4-6',
      'usage': {'prompt_tokens': 0,
       'completion_tokens': 31,
       'total_tokens': 31,
       'cache_read_tokens': 0,
       'cache_creation_tokens': 0},
      'stop_reason': 'aborted',
      'timestamp': 1773456290091}]



## 12. wait_for_idle()

  A coordination primitive. `prompt()` already awaits internally, so when it returns
  the agent is idle. `wait_for_idle()` is for when something *else* triggered the agent
  and you need to sync from a different place in your code:

  ```python
  # Some callback triggered a run
  agent.follow_up("do something")
  asyncio.create_task(agent.continue_run())

  # Later, elsewhere:
  await agent.wait_for_idle()   # block until that run finishes
  # now safe to inspect agent.messages

  If you're always doing await agent.prompt() sequentially, you'll never need this.


```python
# When idle, returns immediately
agent = Agent(model=MODEL)
await agent.wait_for_idle()  # should not hang
print("wait_for_idle() returned immediately (agent is idle)")

# After a prompt, also returns immediately (prompt already blocks)
await agent.prompt("Hi")
await agent.wait_for_idle()
print("wait_for_idle() returned immediately (prompt already completed)")
```

    wait_for_idle() returned immediately (agent is idle)
    wait_for_idle() returned immediately (prompt already completed)



```python
import asyncio

agent = Agent(model=MODEL)

async def background_task():
    agent.follow_up("Summarize what 2+2 is")
    await agent.continue_run()

await agent.prompt("Hi")

# Kick off a run from a background task
asyncio.create_task(background_task())
await asyncio.sleep(0)  # yield to event loop — lets background_task enter _run_loop and set _running_future
await agent.wait_for_idle()

agent.messages
```




    [{'role': 'user', 'content': 'Hi', 'timestamp': 1773456292920},
     {'role': 'assistant',
      'content': 'Hi there! How are you doing? Is there something I can help you with today? 😊',
      'tool_calls': None,
      'thinking_blocks': None,
      'reasoning_content': None,
      'provider_specific_fields': None,
      'model': 'anthropic/claude-sonnet-4-6',
      'usage': {'prompt_tokens': 8,
       'completion_tokens': 24,
       'total_tokens': 32,
       'cache_read_tokens': 0,
       'cache_creation_tokens': 0},
      'stop_reason': 'stop',
      'timestamp': 1773456294426},
     {'role': 'user',
      'content': 'Summarize what 2+2 is',
      'timestamp': 1773456294427},
     {'role': 'assistant',
      'content': "Sure! \n\n**2 + 2 = 4**\n\nSimply put, when you add 2 and 2 together, you get **4**. It's one of the most basic addition facts in mathematics! 😊",
      'tool_calls': None,
      'thinking_blocks': None,
      'reasoning_content': None,
      'provider_specific_fields': None,
      'model': 'anthropic/claude-sonnet-4-6',
      'usage': {'prompt_tokens': 45,
       'completion_tokens': 55,
       'total_tokens': 100,
       'cache_read_tokens': 0,
       'cache_creation_tokens': 0},
      'stop_reason': 'stop',
      'timestamp': 1773456296237}]



## 13. reset() vs clear_messages()

Two ways to clear state, with different scopes:

| Method | Clears messages | Clears queues | Clears error | Keeps config |
|--------|:-:|:-:|:-:|:-:|
| `reset()` | ✓ | ✓ | ✓ | ✓ |
| `clear_messages()` | ✓ | ✗ | ✗ | ✓ |

`reset()` is "start over". `clear_messages()` is "clear history but keep queued work".


```python
agent = Agent(
    model=MODEL,
    system_prompt="you are a nice agent",
    tools=[echo_tool],
)
agent.append_message({"role": "user", "content": "old message"})
agent.steer("queued steering")
agent.follow_up("queued follow-up")
agent._state.error = "some error"

print("Before clear_messages():")
agent.messages
```

    Before clear_messages():





    [{'role': 'user', 'content': 'old message'}]




```python
agent.clear_messages()

print("After clear_messages():")
print(f"  messages: {agent.messages}")
print(f"  queued: {agent.has_queued_messages()}")  # True — queues survived
print(f"  error: {agent.state.error}")              # "some error" — survived
```

    After clear_messages():
      messages: []
      queued: True
      error: some error



```python
# Now reset — clears everything
agent.append_message({"role": "user", "content": "new message"})
agent.reset()

print("After reset():")
print(
    f"  messages: {len(agent.messages)}, queued: {agent.has_queued_messages()}, error: {agent.state.error}"
)
print(f"  model: {agent.state.model}  ← preserved")
print(f"  system_prompt: '{agent.state.system_prompt}'  ← preserved")
print(f"  tools: {len(agent.state.tools)}  ← preserved")
```

    After reset():
      messages: 0, queued: False, error: None
      model: anthropic/claude-sonnet-4-6  ← preserved
      system_prompt: 'you are a nice agent'  ← preserved
      tools: 1  ← preserved


## 14. Configuration setters — mid-run changes

Pi allows calling `setModel()`, `setTools()`, etc. even while the agent is streaming.
The loop snapshots context at the start of each run, so mid-run changes only take
effect on the **next** run. We match this behavior.

This enables patterns like:
- **Dynamic capabilities:** escalate tool permissions mid-conversation
- **Model switching:** use a cheap model for simple tasks, expensive for complex ones
- **Plan mode:** swap tool sets between planning and execution phases


```python
agent = Agent(model=MODEL)


# Track which model gets called
def event_handler(e):
    if e["type"] == "message_end":
        print(f"Model Used: {agent.state.model}")


agent.subscribe(event_handler)

await agent.prompt("hey")

# Switch model mid-conversation
agent.set_model("gemini/gemini-3-flash-preview")

await agent.prompt("Bye")
agent.messages
```

    Model Used: anthropic/claude-sonnet-4-6
    Model Used: anthropic/claude-sonnet-4-6
    Model Used: gemini/gemini-3-flash-preview
    Model Used: gemini/gemini-3-flash-preview





    [{'role': 'user', 'content': 'hey', 'timestamp': 1773456296257},
     {'role': 'assistant',
      'content': "Hey! How's it going? What's on your mind? 😊",
      'tool_calls': None,
      'thinking_blocks': None,
      'reasoning_content': None,
      'provider_specific_fields': None,
      'model': 'anthropic/claude-sonnet-4-6',
      'usage': {'prompt_tokens': 8,
       'completion_tokens': 19,
       'total_tokens': 27,
       'cache_read_tokens': 0,
       'cache_creation_tokens': 0},
      'stop_reason': 'stop',
      'timestamp': 1773456298182},
     {'role': 'user', 'content': 'Bye', 'timestamp': 1773456298182},
     {'role': 'assistant',
      'content': 'Goodbye! Have a great rest of your day. Feel free to reach out whenever you want to chat again! 👋😊',
      'tool_calls': None,
      'thinking_blocks': None,
      'reasoning_content': None,
      'provider_specific_fields': {'thought_signatures': ['EvgDCvUDAb4+9vuTm+YsGYxloE85ilI4hi5dENbsFhDvvQl1wtdz8MOqKX25hZYNoHpKdpyUj9bORtSkRoIJfww03jEhrwjHup+ZGosYGITCSe15BmoaJ5kslC9ALnCchdl0HindxsM9v8YEtqtfe7RPO/MKuKWGQVvOeHhZBwhGoQfBATMC/NrHgcZyMyL/2V6S2+28NK4jVt8Fr84oZj20pBGykuvzb0CEzM2ahM/XubEDNs+1tkFxU+VAwpd5Wm42jKrQxGci3D+ovcnrUpTt6pFPJcyP2S3H6Ds2cFUKyg+av6csJr4k/z1p8YOj5vdelYQFYWbj0515SvJnHvvPpanbWUIwLtflzanzAGEwiCudl4+gYo0nOrqJtkOeLXMxSTSKl3RUoG+WwKWtzgTADH1WDsKk3qrVvkD0XmIQodUxhAtNyWCVNoYgke7fV2OUbVx0qTd+KgNWb96iZwXw+VJgoujMUpfNqW4Fuyc5BIeqIvHZtQvSurzwzEhEOkEbDaBpKDvjZFCmA7o/QIWITVzxfIoD/G1oKmj/2yFO3cJKAFdDJ+5EUPZ01cidQZIpbVUXegrNyg4FLOwP1JkRwcPvYR1bgMC4PzTL0T8TphPnSbcExtdCwjzmivaqp2Ma9aVDJpdMn1U5GtNcBxglYRfWqV5BCCpj']},
      'model': 'gemini/gemini-3-flash-preview',
      'usage': {'prompt_tokens': 21,
       'completion_tokens': 130,
       'total_tokens': 151,
       'cache_read_tokens': 0,
       'cache_creation_tokens': 0},
      'stop_reason': 'stop',
      'timestamp': 1773456300020}]



## 15. Error handling

When the LLM call fails (network error, rate limit, etc.), the Agent:
1. Catches the exception
2. Creates a synthetic assistant message with `stop_reason="error"`
3. Appends it to messages
4. Sets `agent.state.error`
5. Emits `agent_end` event
6. Cleans up (is_streaming=False, etc.)

The Agent does **not** re-raise — it always completes cleanly. This lets consumers
check `agent.state.error` instead of wrapping every `prompt()` in try/except.


```python
# Force an error by using a non-existent model
agent = Agent(model="fake-provider/nonexistent-model")
await agent.prompt("This will fail.")

agent.messages
```

    
    [1;31mProvider List: https://docs.litellm.ai/docs/providers[0m
    
    
    [1;31mProvider List: https://docs.litellm.ai/docs/providers[0m
    





    [{'role': 'user', 'content': 'This will fail.', 'timestamp': 1773456304619},
     {'role': 'assistant',
      'content': None,
      'tool_calls': None,
      'thinking_blocks': None,
      'reasoning_content': None,
      'provider_specific_fields': None,
      'model': 'fake-provider/nonexistent-model',
      'usage': {'prompt_tokens': 0,
       'completion_tokens': 0,
       'total_tokens': 0,
       'cache_read_tokens': 0,
       'cache_creation_tokens': 0},
      'stop_reason': 'error',
      'timestamp': 1773456304623,
      'error_message': "litellm.BadRequestError: LLM Provider NOT provided. Pass in the LLM provider you are trying to call. You passed model=fake-provider/nonexistent-model\n Pass model as E.g. For 'Huggingface' inference endpoints pass in `completion(model='huggingface/starcoder',..)` Learn more: https://docs.litellm.ai/docs/providers"}]



When a tool raises, the exception is caught and turned into a `ToolResult` with `is_error=True`. The error gets sent back to the LLM as a tool result, and the loop continues — the LLM sees the error and can recover or report it. The run doesn't abort.


```python

async def failing_fn(tool_call_id, args, signal, on_update):
    raise ValueError("Something went wrong!")

failing_tool = Tool(
    name="failing_tool",
    description="A tool that always fails",
    parameters={"type": "object", "properties": {}},
    execute=failing_fn,
)

agent = Agent(model=MODEL, tools=[failing_tool], system_prompt="Use the failing_tool when asked.")
await agent.prompt("Use your tool")
agent.messages
```




    [{'role': 'user', 'content': 'Use your tool', 'timestamp': 1773456308222},
     {'role': 'assistant',
      'content': 'Sure! Let me use the tool right away.',
      'tool_calls': [{'id': 'toolu_019S3GNA98PaGZctFMy3p2Qk',
        'type': 'function',
        'function': {'name': 'failing_tool', 'arguments': '{}'},
        'provider_specific_fields': None}],
      'thinking_blocks': None,
      'reasoning_content': None,
      'provider_specific_fields': None,
      'model': 'anthropic/claude-sonnet-4-6',
      'usage': {'prompt_tokens': 555,
       'completion_tokens': 47,
       'total_tokens': 602,
       'cache_read_tokens': 0,
       'cache_creation_tokens': 0},
      'stop_reason': 'tool_calls',
      'timestamp': 1773456309847},
     {'role': 'tool',
      'tool_call_id': 'toolu_019S3GNA98PaGZctFMy3p2Qk',
      'name': 'failing_tool',
      'content': [{'type': 'text', 'text': 'Something went wrong!'}],
      'details': {},
      'is_error': True,
      'timestamp': 1773456309847},
     {'role': 'assistant',
      'content': 'As expected, the tool failed with the error: **"Something went wrong!"** — This tool is designed to always fail, so this is the expected behavior. Let me know if there\'s anything else I can help you with!',
      'tool_calls': None,
      'thinking_blocks': None,
      'reasoning_content': None,
      'provider_specific_fields': None,
      'model': 'anthropic/claude-sonnet-4-6',
      'usage': {'prompt_tokens': 617,
       'completion_tokens': 50,
       'total_tokens': 667,
       'cache_read_tokens': 0,
       'cache_creation_tokens': 0},
      'stop_reason': 'stop',
      'timestamp': 1773456311519}]



## 16. make_default_convert — the default converter

If you don't provide `convert_to_llm`, the Agent builds one via
`make_default_convert(model)` from `liteagent/convert.py`.

This is the **sole provider-specific boundary** in the codebase. It uses a
**denylist** approach. It strips known `liteagent` metadata fields, pass everything
else through. 

**What it strips:** `timestamp`, `usage`, `stop_reason`, `error_message`,
`details`, `is_error`

**What it preserves:** `thinking_blocks`, `reasoning_content`,
`provider_specific_fields`, multimodal content blocks, tool call metadata,
everything the LLM needs for multi-turn functionality.

**OpenAI image hoisting:** For OpenAI models, tool results with `[text, image_url]`
content get split — text stays in the tool message, images are hoisted into a
synthetic user message. This is because OpenAI's Chat Completions API silently
ignores image blocks in tool result content. Anthropic and Gemini handle them
natively, so no hoisting needed.

It's not perfect. There are many quirks and edge cases with `litellm`.
I will evolve this over time.


```python
from liteagent.convert import make_default_convert

# Build the converter for Anthropic (no image hoisting needed)
convert_anthropic = make_default_convert("anthropic/claude-sonnet-4-6")

# Build one for OpenAI (will hoist tool-result images)
convert_openai = make_default_convert("gpt-5.2")

# Simulate a conversation with enriched messages
messages = [
    {"role": "user", "content": "Hi", "timestamp": 12345},
    {
        "role": "assistant",
        "content": "Hello!",
        "tool_calls": None,
        "thinking_blocks": [{"type": "thinking", "thinking": "greeting"}],
        "reasoning_content": "simple greeting",
        "provider_specific_fields": {"thought_signatures": ["sig123"]},
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        "stop_reason": "stop",
        "timestamp": 12346,
    },
    {
        "role": "tool",
        "tool_call_id": "c0",
        "name": "chart",
        "content": [
            {"type": "text", "text": "Here is the chart."},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
        ],
        "is_error": False,
        "details": {"extra": "ui-only"},
        "timestamp": 12347,
    },
]

print("=== Anthropic converter (images stay in tool message) ===")
for m in convert_anthropic(messages):
    print(m)

print()
print("=== OpenAI converter (images hoisted to user message) ===")
for m in convert_openai(messages):
    print(m)
```

    === Anthropic converter (images stay in tool message) ===
    {'role': 'user', 'content': 'Hi'}
    {'role': 'assistant', 'content': 'Hello!', 'tool_calls': None, 'thinking_blocks': [{'type': 'thinking', 'thinking': 'greeting'}], 'reasoning_content': 'simple greeting', 'provider_specific_fields': {'thought_signatures': ['sig123']}}
    {'role': 'tool', 'tool_call_id': 'c0', 'name': 'chart', 'content': [{'type': 'text', 'text': 'Here is the chart.'}, {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,abc'}}]}
    
    === OpenAI converter (images hoisted to user message) ===
    {'role': 'user', 'content': 'Hi'}
    {'role': 'assistant', 'content': 'Hello!', 'tool_calls': None, 'thinking_blocks': [{'type': 'thinking', 'thinking': 'greeting'}], 'reasoning_content': 'simple greeting', 'provider_specific_fields': {'thought_signatures': ['sig123']}}
    {'role': 'tool', 'tool_call_id': 'c0', 'name': 'chart', 'content': 'Here is the chart.'}
    {'role': 'user', 'content': [{'type': 'text', 'text': 'Image from tool result:'}, {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,abc'}}]}


The default converter handles the common cases transparently. You'd override it when:

1. **Custom message types** — your app stores `{"role": "notification", ...}` that need
   to be filtered or converted
2. **Custom compaction** — you want to summarize old messages before sending
3. **Provider-specific optimizations** — e.g., different image handling for a specific model

For many uses, the default just works and you won't need to pass `convert_to_llm`.

## 17. Testing across models

The Agent is model-agnostic. Let's run the same prompt + tool call through some different models.


```python
MODELS = [
    "anthropic/claude-sonnet-4-6",
    "anthropic/claude-opus-4-6",
    "gemini/gemini-3-pro-preview",
    "gemini/gemini-3-flash-preview",
    "gpt-5.2",
    "gpt-5.3-codex",
    "gpt-5.4",
]

for model in MODELS:
    agent = Agent(
        model=model,
        system_prompt="Use the echo tool. Be concise.",
        tools=[echo_tool],
    )

    try:
        await agent.prompt("Echo 'test'")
        roles = [m.get("role") for m in agent.messages]
        has_tool = "tool" in roles
        assistants = [m for m in agent.messages if m.get("role") == "assistant"]
        last_content = (assistants[-1].get("content") or "")[:50] if assistants else "?"
        print(f"  ✓ {model:<42} tool_used={has_tool} | {last_content}")
    except Exception as e:
        print(f"  ✗ {model:<42} ERROR: {e}")
```

      ✓ anthropic/claude-sonnet-4-6                tool_used=True | The echoed message is: **test**
      ✓ anthropic/claude-opus-4-6                  tool_used=True | Done! The message "test" was echoed back.
      ✓ gemini/gemini-3-pro-preview                tool_used=True | You're welcome!
      ✓ gemini/gemini-3-flash-preview              tool_used=True | test
      ✓ gpt-5.2                                    tool_used=True | 'test'


    /Users/christopher/personal_projects/DrChrisLevy.github.io/.venv/lib/python3.11/site-packages/pydantic/main.py:464: UserWarning: Pydantic serializer warnings:
      PydanticSerializationUnexpectedValue(Expected `ResponseAPIUsage` - serialized value may not be as expected [field_name='usage', input_value={'completion_tokens': 17,..., 'video_tokens': None}}, input_type=dict])
      return self.__pydantic_serializer__.to_python(
    /Users/christopher/personal_projects/DrChrisLevy.github.io/.venv/lib/python3.11/site-packages/pydantic/main.py:464: UserWarning: Pydantic serializer warnings:
      PydanticSerializationUnexpectedValue(Expected `ResponseAPIUsage` - serialized value may not be as expected [field_name='usage', input_value={'completion_tokens': 5, ..., 'video_tokens': None}}, input_type=dict])
      return self.__pydantic_serializer__.to_python(


      ✓ gpt-5.3-codex                              tool_used=True | test
      ✓ gpt-5.4                                    tool_used=True | test


## 18. Real-world patterns

The Agent is framework-agnostic. Here's how you'd wire it into different consumers.


```python
# Pattern 1: CLI — print text deltas as they arrive

agent = Agent(
    model=MODEL,
    system_prompt="Be concise.",
)


def cli_handler(event):
    if event["type"] == "message_update" and event.get("delta_type") == "text_delta":
        text = event["delta"].get("content", "")
        print(text, end="", flush=True)
    elif event["type"] == "agent_end":
        print()  # newline at the end


agent.subscribe(cli_handler)
print("Agent: ", end="")
await agent.prompt("What is the meaning of life, in one sentence?")
```

    Agent: The meaning of life is whatever purpose, connection, and fulfillment you consciously choose to create and pursue.


## 19. Multimodal tool results across providers

The default converter handles multimodal tool results transparently:
- **Anthropic / Gemini**: images stay in the tool message (native support)
- **OpenAI**: images are hoisted into a synthetic user message (OpenAI ignores image blocks in tool results)

This test sends a bar chart image to the model via a tool result. The chart has
an obvious spike in May (580 vs ~130 baseline). The model must identify "May"
from the image — proving it actually saw the image, not just the text.

This exercises the full round-trip: tool returns `[text, image_url]` →
default converter handles it per-provider → model reasons about the image.




```python
import base64
import io


def make_bar_chart_b64():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun"]
    values = [120, 135, 128, 142, 580, 131]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(months, values, color=["#3498db" if v < 300 else "#e74c3c" for v in values])
    ax.set_title("Monthly API Errors")
    for i, v in enumerate(values):
        ax.text(i, v + 15, str(v), ha="center", fontweight="bold")
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=72)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


chart_b64 = make_bar_chart_b64()


async def get_chart_exec(tool_call_id, params, signal=None, on_update=None):
    return ToolResult(
        content=[
            {"type": "text", "text": "Here is the monthly error chart."},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{chart_b64}"},
            },
        ],
    )


chart_tool = Tool(
    name="get_error_chart",
    description="Get the monthly error chart. Returns text + chart image.",
    parameters={"type": "object", "properties": {}},
    execute=get_chart_exec,
)

MULTIMODAL_MODELS = [
    "anthropic/claude-sonnet-4-6",
    "anthropic/claude-opus-4-6",
    "gpt-5.2",
    "gpt-5.3-codex",
    "gpt-5.4",
]

for model in MULTIMODAL_MODELS:
    agent = Agent(
        model=model,
        tools=[chart_tool],
        system_prompt=(
            "Use get_error_chart when asked. After seeing the chart, "
            "answer the user's question about it. Be concise."
        ),
    )
    try:
        await agent.prompt(
            "Get the error chart, then tell me: which month has the highest "
            "error count? Reply with just the month name."
        )
        assistants = [m for m in agent.messages if m.get("role") == "assistant"]
        last = assistants[-1] if assistants else {}
        if last.get("error_message"):
            raise RuntimeError(last["error_message"])
        answer = (last.get("content") or "").lower()
        saw_image = "may" in answer
        status = "PASSED" if saw_image else "FAILED"
        print(f"  {status} {model:<42} | {answer[:60]}")
    except Exception as e:
        print(f"  FAILED {model:<42} | ERROR: {e}")
```

      PASSED anthropic/claude-sonnet-4-6                | may
      PASSED anthropic/claude-opus-4-6                  | **may**
      PASSED gpt-5.2                                    | may


    /Users/christopher/personal_projects/DrChrisLevy.github.io/.venv/lib/python3.11/site-packages/pydantic/main.py:464: UserWarning: Pydantic serializer warnings:
      PydanticSerializationUnexpectedValue(Expected `ResponseAPIUsage` - serialized value may not be as expected [field_name='usage', input_value={'completion_tokens': 15,..., 'video_tokens': None}}, input_type=dict])
      return self.__pydantic_serializer__.to_python(


      PASSED gpt-5.3-codex                              | may
      PASSED gpt-5.4                                    | may


# Conclusion

This wraps up my intro three part series on porting [pi-mono](https://github.com/badlogic/pi-mono)'s agent loop to Python.

- [Part 1](https://drchrislevy.com/blog/blog_post?fpath=posts%2Fasync_event_streams%2Fasync_event_streams.ipynb) built the async event stream. This is the producer-consumer queue that everything flows through.
- [Part 2](https://drchrislevy.com/blog/blog_post?fpath=posts%2Fdual_loop%2Fexploring-loop.ipynb) built the raw loop — the stateless dual while-loop that calls the LLM, executes tools, and handles steering/follow-ups.
- **Part 3** (this post) wrapped it with the `Agent` class. That is the stateful layer that manages message history, event subscription, queues, and cancellation.

The result is [liteagent](https://github.com/DrChrisLevy/liteagent).

## The litellm trade-off

The biggest design decision in liteagent is delegating all provider communication to [litellm](https://github.com/BerriAI/litellm) instead of writing custom provider code.

pi-mono's `packages/ai/src/providers/` contains  thousands of lines TypeScript across files for providers like Anthropic, OpenAI Completions, OpenAI Codex Responses, Google, Google Vertex, Google Gemini CLI, Mistral, Amazon Bedrock, etc.

`liteagent`'s entire provider boundary is less than 100 lines in a single file (`convert.py`). It strips `liteagent` metadata, passes everything else through, and handles one provider quirk: hoisting images from tool results into synthetic user messages for OpenAI (which silently ignores image blocks in tool result content).

It's not perfect though.The upside is obvious, less to write, less to maintain, access to every model litellm supports. The downside is you inherit every litellm bug and inconsistency, and you can't fix them at the source easily. I can see why pi-mono
went the custom route, and honestly I'm tempted at times to do the same for liteagent.

## litellm issues encountered during this project

Many of the issues are documented in [DESIGN_NOTES](https://github.com/DrChrisLevy/liteagent/blob/main/DESIGN_NOTES.md) and [LITELLM_API_LANDSCAPE](https://github.com/DrChrisLevy/liteagent/blob/main/learnings/LITELLM_API_LANDSCAPE.md). I think litellm is a great project, and has a very difficult task of mapping out all the llm providers patterns and unifying it all. A few of the issues I ran into:


**Thinking metadata is inconsistent across providers.** Anthropic gets `reasoning_content` (string) plus first-class `thinking_blocks` (with cryptographic signatures). Gemini gets `reasoning_content` (sometimes — absent on trivial prompts) with signatures buried in `provider_specific_fields["thought_signatures"]`. OpenAI's GPT-5.x thinking is completely invisible through Chat Completions (the model spends reasoning tokens but the content is hidden).

**GPT-5.4 reasoning is silently disabled with tools.** litellm quietly drops `reasoning_effort` when tools are present for GPT-5.4 specifically. This only affects 5.4; earlier models handle reasoning + tools fine through Chat Completions. The Responses API doesn't have this limitation, but litellm's `acompletion()` doesn't use it for base GPT models.

**OpenAI ignores images in tool results.** OpenAI's Chat Completions API accepts only string content in tool messages. If you send `[text, image_url]` blocks, the model sees the text but the image is silently dropped. Anthropic and Gemini handle multimodal tool results natively. This required a provider-specific workaround in the converter, the one place liteagent has provider-aware code.

The key insight from this port is that the agent loop itself is provider-agnostic. The dual while-loop, steering, follow-ups, cancellation, event streaming, etc. do not
depend on the provider. None of that cares which LLM is on the other end. The provider pain is entirely at the boundary: message format conversion, thinking metadata, multimodal content handling. Isolating that boundary to a single file (`convert.py`) keeps the core clean even when litellm forces workarounds.

But I'm still on the fence about the litellm trade-off. I want to spend some time building consumers that use liteagent, and see how much pain I actually have to deal with.

## Next steps

`pi-mono` has so many great patterns. I love the streaming of events and having consumers subscribing to them. I also think the dual loop architecture with steering and follow-ups is a great pattern too. It goes beyond just a standard agent loop with some
simple yet elegant "bells and whistles". I'm excited to build some tooling and custom agents around my python liteagent implementation.
