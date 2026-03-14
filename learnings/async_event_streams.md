---
title: Part 1 of Porting pi-mono's agent loop to Python - Async and Event Streams
author: Chris Levy
draft: false
date: '2026-03-07'
date-modified: '2026-03-07'
image: async_event_streams.jpg
description: "Part 1 of porting pi-mono's agent loop to Python. Building up async concepts from scratch — coroutines, queues, futures, and async iteration — ending with liteagent's EventStream"
tags:
  - python
  - async
  - agents
---
# Intro

This is part 1 of a series where I port [pi-mono](https://github.com/badlogic/pi-mono)'s agent loop from TypeScript to Python as a learning exercise. The result is [liteagent](https://github.com/DrChrisLevy/liteagent), which is still a wip.

The series follows the same bottom-up structure as the codebase:

1. **EventStream** (this post) — async primitives and the streaming foundation
2. **The Agent Loop** — LLM calls, tool execution, and the core loop
3. **The Agent** — tying it all together with configuration and steering

This post builds up from basic coroutines to `asyncio.Queue`, `Future`, and `async for`, ending with `liteagent`'s `EventStream`, the core abstraction that connects the agent loop to any consumer (CLI, web UI, etc.).

## What problem are we solving?

Our agent loop does work (calls an LLM, runs tools, etc.) and we want to
show tokens appearing word by word, tool execution progress, etc.

The naive approach blocks until everything is done:
```python
result = do_all_the_work()
print(result)
```

What we want:
```python
for event in stream_of_events:
    print(event)  # each event arrives as it happens
```

But the work (producing events) and the display (consuming events) need to
happen at the same time. The producer can't wait for the consumer, and
the consumer can't rush the producer.

This is the **producer-consumer pattern**.

## What is async?

Async lets one Python thread have multiple things **waiting** at the same time.

**Async is not parallel.** Only one line of Python runs at any given instant.
Nothing runs "at the same time" in the CPU sense. That's `multiprocessing`.

| Approach | What it does |
|---|---|
| `asyncio` | One thread. Overlaps **waiting** (network, disk, sleep). |
| `threading` | Multiple threads. Some concurrency, but limited by Python's GIL. |
| `multiprocessing` | Multiple processes. Truly parallel on separate CPU cores. |


### Sync vs async


```python
# SYNC — one after the other, total: ~1.5 seconds
import time


def make_coffee_sync():
    print("start brewing")
    time.sleep(1)
    print("coffee ready")


def make_toast_sync():
    print("start toasting")
    time.sleep(0.5)
    print("toast ready")


start = time.time()
make_coffee_sync()
make_toast_sync()
print(f"total: {time.time() - start:.1f}s")
```

    start brewing
    coffee ready
    start toasting
    toast ready
    total: 1.5s



```python
# ASYNC — waits overlap, total: ~1.0 seconds
import asyncio


async def make_coffee():
    print("start brewing")
    await asyncio.sleep(1)
    print("coffee ready")


async def make_toast():
    print("start toasting")
    await asyncio.sleep(0.5)
    print("toast ready")


start = time.time()
await asyncio.gather(make_coffee(), make_toast())
print(f"total: {time.time() - start:.1f}s")
```

    start brewing
    start toasting
    toast ready
    coffee ready
    total: 1.0s


The time savings come from overlapping the **waiting**, not from running
Python code simultaneously.

Async only helps when your code spends time waiting for things **outside Python**. Things such as
network calls, database queries, file I/O, subprocess results, timers.
If your code is pure computation, async gives you zero benefit.

Our agent loop is almost all I/O waiting (LLM API calls, tool execution),
which is asyncio's sweet spot.

##  async def,  await

```python
async def fetch_user():     # defines an async function
    ...

fetch_user()                # creates a "coroutine" object — DOESN'T RUN IT

await fetch_user()          # actually runs the function
```

`await` = run this async function. You must use `await` to call async
functions, just like you use `()` to call regular functions.

Most code inside an `async def` is regular Python. You only need `await`
when calling something that's itself `async def`:

```python
async def process_user():
    response = await fetch_user()   # async — needs await
    name = response["name"]         # regular Python
    upper = name.upper()            # regular Python
    await save_to_db(result)        # async — needs await
    return result
```

## Running multiple things concurrently

`await` runs ONE thing and waits for it. To overlap multiple things,
you need `gather` or `create_task`.


```python
# Sequential — 1.5 seconds total
start = time.time()
await asyncio.sleep(1)
await asyncio.sleep(0.5)
print(f"sequential: {time.time() - start:.1f}s")

# Concurrent — 1.0 second total (waits overlap)
start = time.time()
await asyncio.gather(asyncio.sleep(1), asyncio.sleep(0.5))
print(f"concurrent: {time.time() - start:.1f}s")
```

    sequential: 1.5s
    concurrent: 1.0s


### How switching works

Only one coroutine runs at a time. Every `await` is a **switch point** —
Python can pause this coroutine and resume another one.


```python
# Watch the interleaving
await asyncio.gather(make_coffee(), make_toast())
# make_coffee: "start brewing"    <- runs
# make_coffee: await sleep(1)     <- pauses, python checks: anyone else?
# make_toast:  "start toasting"   <- runs (coffee is sleeping)
# make_toast:  await sleep(0.5)   <- pauses, both sleeping now
#              ... 0.5s pass ...
# make_toast:  "toast ready"      <- wakes up
#              ... 0.5s more ...
# make_coffee: "coffee ready"     <- wakes up
```

    start brewing
    start toasting
    toast ready
    coffee ready





    [None, None]



### What about pure computation?

If an async function does heavy computation between `await`s, it blocks
everything else during that time. There's nowhere to switch.


```python
def heavy_math(n):
    """Pure computation — no await points, blocks the event loop."""
    total = 0
    for i in range(n):
        total += i * i
    return total


async def crunch():
    print("  loading...")
    await asyncio.sleep(0.1)  # switch point
    print("  crunching (blocks everything)...")
    result = heavy_math(5_000_000)  # NO switching possible
    print(f"  done: {result:,}")


start = time.time()
await asyncio.gather(crunch(), crunch())
print(f"total: {time.time() - start:.1f}s (loads overlap, crunches don't)")
```

      loading...
      loading...
      crunching (blocks everything)...
      done: 41,666,654,166,667,500,000
      crunching (blocks everything)...
      done: 41,666,654,166,667,500,000
    total: 0.3s (loads overlap, crunches don't)


## asyncio.Queue

A queue lets one coroutine put stuff in, and another take stuff out.
`asyncio.Queue` is async-aware. The consumer can `await` the next item
while the producer keeps working.

- `put_nowait()` — shove it in immediately (our queue has no max size)
- `await queue.get()` — waits if the queue is empty. While it's paused,
  the producer can run and push more stuff.


```python
async def producer(queue):
    for i in range(5):
        await asyncio.sleep(0.3)
        queue.put_nowait(f"event {i}")
        print(f"  [producer] pushed event {i}")
    queue.put_nowait(None)  # signal: "I'm done"


async def consumer(queue):
    while True:
        event = await queue.get()
        if event is None:
            break
        print(f"  [consumer] got: {event}")


queue = asyncio.Queue()
await asyncio.gather(producer(queue), consumer(queue))
```

      [producer] pushed event 0
      [consumer] got: event 0
      [producer] pushed event 1
      [consumer] got: event 1
      [producer] pushed event 2
      [consumer] got: event 2
      [producer] pushed event 3
      [consumer] got: event 3
      [producer] pushed event 4
      [consumer] got: event 4





    [None, None]



### What happens when the consumer is slower than the producer?

Events buffer in the queue. The queue grows.


```python
async def fast_producer(queue):
    for i in range(5):
        await asyncio.sleep(0.1)  # fast
        queue.put_nowait(i)
        print(f"  [producer] pushed {i} (queue size: {queue.qsize()})")
    queue.put_nowait(None)


async def slow_consumer(queue):
    while True:
        event = await queue.get()
        if event is None:
            break
        print(f"  [consumer] processing {event}...")
        await asyncio.sleep(0.5)  # slow
    print("  [consumer] done")


queue = asyncio.Queue()
await asyncio.gather(fast_producer(queue), slow_consumer(queue))
```

      [producer] pushed 0 (queue size: 1)
      [consumer] processing 0...
      [producer] pushed 1 (queue size: 1)
      [producer] pushed 2 (queue size: 2)
      [producer] pushed 3 (queue size: 3)
      [producer] pushed 4 (queue size: 4)
      [consumer] processing 1...
      [consumer] processing 2...
      [consumer] processing 3...
      [consumer] processing 4...
      [consumer] done





    [None, None]



## asyncio.Future

A Future is a box that starts empty and gets filled exactly once.

```python
future = asyncio.get_running_loop().create_future()

# Somewhere later:
future.set_result("the answer")    # fills the box (once only)

# Someone waiting:
result = await future              # waits until the box is filled
```

We use this for `stream.result()` — the consumer can await the final result
of the entire stream.


```python
async def fill_later(future):
    print("working...")
    await asyncio.sleep(1)
    future.set_result(42)
    print("future filled!")


future = asyncio.get_running_loop().create_future()
task = asyncio.create_task(fill_later(future))
result = await future
print(f"got: {result}")
```

    working...
    future filled!
    got: 42


## async for

If a class defines `__aiter__`, you can use `async for`. The key difference
from regular `for`: **getting the next item** might require waiting.

```python
# Regular for: getting the next item is instant
for item in [1, 2, 3]:
    ...

# Async for: getting the next item requires waiting
async for event in stream:
    ...  # __anext__() awaits — waits for the next event to arrive
```

`async for` is the consumer pattern in nice syntax. It sits there
waiting for the producer to push the next item, processes it, waits again.


```python
# async generator — yield inside async def
async def countdown(n):
    for i in range(n, 0, -1):
        await asyncio.sleep(0.3)
        yield i


async for num in countdown(3):
    print(num)
```

    3
    2
    1



```python
# async for + gather = interleaved coroutines
async def heartbeat():
    for _ in range(5):
        await asyncio.sleep(0.2)
        print("  heartbeat")


async def run_countdown():
    async for num in countdown(3):
        print(f"  countdown: {num}")


await asyncio.gather(run_countdown(), heartbeat())
```

      heartbeat
      countdown: 3
      heartbeat
      countdown: 2
      heartbeat
      heartbeat
      countdown: 1
      heartbeat





    [None, None]



## EventStream

Now we have all the pieces:

- **`asyncio.Queue`** — buffer events between producer and consumer
- **`asyncio.Future`** — hold the final result
- **`async for` / `__aiter__`** — let consumers iterate naturally
- **Sentinel value** — signal "stream is done" through the queue

`liteagent`'s `EventStream` combines all four:


```python
from IPython.display import Markdown, display
import inspect
from liteagent.stream import EventStream



source = inspect.getsource(EventStream)
display(Markdown(f"```python\n{source}\n```"))
```


```python
class EventStream:
    """Async event stream. Producer pushes events, consumer iterates with `async for`."""

    def __init__(self):
        self._queue: asyncio.Queue = asyncio.Queue()
        self._done = False
        self._result_future: asyncio.Future = asyncio.get_running_loop().create_future()

    def push(self, event):
        if self._done:
            return
        self._queue.put_nowait(event)

    def end(self, result=None):
        if self._done:
            return
        self._done = True
        if not self._result_future.done():
            self._result_future.set_result(result)
        self._queue.put_nowait(_SENTINEL)

    async def result(self):
        return await self._result_future

    async def __aiter__(self):
        while True:
            event = await self._queue.get()
            if event is _SENTINEL:
                return
            yield event

```


### Basic usage: push events, iterate them


```python
stream = EventStream()

# Push events first, THEN iterate (events buffer in queue)
stream.push({"type": "hello", "text": "first event"})
stream.push({"type": "hello", "text": "second event"})
stream.end("done!")

async for event in stream:
    print(f"got: {event}")

result = await stream.result()
print(f"result: {result}")
```

    got: {'type': 'hello', 'text': 'first event'}
    got: {'type': 'hello', 'text': 'second event'}
    result: done!


### Producer and consumer running concurrently


```python
stream = EventStream()


async def es_producer():
    for i in range(5):
        await asyncio.sleep(0.3)
        stream.push({"type": "update", "n": i})
        print(f"  [producer] pushed {i}")
    stream.end({"total_events": 5})
    print("  [producer] ended stream")


async def es_consumer():
    async for event in stream:
        print(f"  [consumer] got {event}")
        await asyncio.sleep(0.1)
    print(f"  [consumer] result = {await stream.result()}")


await asyncio.gather(es_producer(), es_consumer())
```

      [producer] pushed 0
      [consumer] got {'type': 'update', 'n': 0}
      [producer] pushed 1
      [consumer] got {'type': 'update', 'n': 1}
      [producer] pushed 2
      [consumer] got {'type': 'update', 'n': 2}
      [producer] pushed 3
      [consumer] got {'type': 'update', 'n': 3}
      [producer] pushed 4
      [producer] ended stream
      [consumer] got {'type': 'update', 'n': 4}
      [consumer] result = {'total_events': 5}





    [None, None]



### Awaiting just the result (skip the stream)


```python
stream = EventStream()


async def background_work():
    stream.push({"type": "working"})
    await asyncio.sleep(0.5)
    stream.push({"type": "still working"})
    await asyncio.sleep(0.5)
    stream.end(["message1", "message2"])


# schedule producer, then just await the final result
asyncio.create_task(background_work())
result = await stream.result()
print(f"final result: {result}")
```

    final result: ['message1', 'message2']


## How this fits into the agent

The agent loop runs as a background task (via `asyncio.create_task`).
The consumer iterates the stream. They run concurrently. The producer
pushes events whenever it has them, and the consumer processes them
whenever it's ready. The queue bridges the gap.

Let's simulate it:


```python
stream = EventStream()
all_messages = []


async def fake_agent_loop():
    """Simulates the real agent loop."""
    stream.push({"type": "agent_start"})

    # User message
    user_msg = {"role": "user", "content": "hello"}
    all_messages.append(user_msg)
    stream.push({"type": "message_start", "message": user_msg})
    stream.push({"type": "message_end", "message": user_msg})

    # LLM streams a response
    assistant_msg = {"role": "assistant", "content": ""}
    stream.push({"type": "message_start", "message": assistant_msg})

    for word in ["Hello", " there", "!", " How", " can", " I", " help", "?"]:
        await asyncio.sleep(0.1)
        assistant_msg["content"] += word
        stream.push(
            {"type": "message_update", "delta_type": "text_delta", "delta": word}
        )

    all_messages.append(assistant_msg)
    stream.push({"type": "message_end", "message": assistant_msg})
    stream.push({"type": "agent_end", "messages": all_messages})
    stream.end(all_messages)


# Schedule the loop, then consume events — just like a real CLI
asyncio.create_task(fake_agent_loop())

async for event in stream:
    print(event)
```

    {'type': 'agent_start'}
    {'type': 'message_start', 'message': {'role': 'user', 'content': 'hello'}}
    {'type': 'message_end', 'message': {'role': 'user', 'content': 'hello'}}
    {'type': 'message_start', 'message': {'role': 'assistant', 'content': ''}}
    {'type': 'message_update', 'delta_type': 'text_delta', 'delta': 'Hello'}
    {'type': 'message_update', 'delta_type': 'text_delta', 'delta': ' there'}
    {'type': 'message_update', 'delta_type': 'text_delta', 'delta': '!'}
    {'type': 'message_update', 'delta_type': 'text_delta', 'delta': ' How'}
    {'type': 'message_update', 'delta_type': 'text_delta', 'delta': ' can'}
    {'type': 'message_update', 'delta_type': 'text_delta', 'delta': ' I'}
    {'type': 'message_update', 'delta_type': 'text_delta', 'delta': ' help'}
    {'type': 'message_update', 'delta_type': 'text_delta', 'delta': '?'}
    {'type': 'message_end', 'message': {'role': 'assistant', 'content': 'Hello there! How can I help?'}}
    {'type': 'agent_end', 'messages': [{'role': 'user', 'content': 'hello'}, {'role': 'assistant', 'content': 'Hello there! How can I help?'}]}

