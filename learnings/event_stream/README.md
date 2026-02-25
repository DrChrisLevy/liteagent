# Learning: Event Streams (from scratch)


## Part 1: What problem are we solving?

Our agent loop does work (calls an LLM, runs tools, etc.) and we want to
**show what's happening in real time** — tokens appearing word by word,
tool execution progress, etc.

The naive approach:

```python
# BAD: blocks until everything is done
result = do_all_the_work()
print(result)
```

What we want:

```python
# GOOD: see events as they happen
for event in stream_of_events:
    print(event)  # each event arrives as it happens
```

But there's a problem: the work (producing events) and the display (consuming
events) need to happen **at the same time**. The producer can't wait for the
consumer, and the consumer can't rush the producer.

This is the **producer-consumer pattern**.

---

## Part 2: What is async?

Async lets one Python thread have multiple things **waiting** at the same time.

### What it is NOT

**Async is not parallel.** Only one line of Python runs at any given instant.
Nothing runs "at the same time" in the CPU sense. That's `multiprocessing`.

| Approach | What it does |
|---|---|
| `asyncio` | One thread. Overlaps **waiting** (network, disk, sleep). |
| `threading` | Multiple threads. Some concurrency, but limited by Python's GIL. |
| `multiprocessing` | Multiple processes. Truly parallel on separate CPU cores. |

### What it IS

**Concurrent waiting.** Like one person with three things in the oven.
Three things are "cooking" at the same time, but you only have two hands —
you can only put one in or take one out at a time.

### Sync vs async in code

```python
# SYNC — one after the other, total: 5 seconds
import time

def make_coffee():
    print("start brewing")
    time.sleep(3)            # blocks everything for 3 seconds
    print("coffee ready")

def make_toast():
    print("start toasting")
    time.sleep(2)            # blocks everything for 2 seconds
    print("toast ready")

make_coffee()
make_toast()
```

```python
# ASYNC — waits overlap, total: 3 seconds
import asyncio

async def make_coffee():
    print("start brewing")
    await asyncio.sleep(3)   # this function waits, but others can run
    print("coffee ready")

async def make_toast():
    print("start toasting")
    await asyncio.sleep(2)   # this function waits, but others can run
    print("toast ready")

async def main():
    await asyncio.gather(make_coffee(), make_toast())

asyncio.run(main())
```

The time savings come from overlapping the **waiting**, not from running
Python code simultaneously.

### When async helps (and when it doesn't)

Async only helps when your code spends time **waiting for things outside
Python** — network calls, database queries, file I/O, subprocess results,
timers.

If your code is pure computation (math, loops, string processing), async
gives you zero benefit. There's no waiting to overlap.

Our agent loop is almost all I/O waiting (LLM API calls, tool execution),
which is asyncio's sweet spot.

---

## Part 3: The three things — `async def`, calling, `await`

### Defining

```python
async def fetch_user():     # defines an async function
    ...
```

Just like `def` defines a regular function. Nothing runs yet.

### Calling (creates a coroutine)

```python
fetch_user()                # creates a "coroutine" object — DOESN'T RUN IT
```

This is like writing a recipe card. No work happens. A "coroutine" is just
the name for this not-yet-run object.

### Awaiting (actually runs it)

```python
await fetch_user()          # actually runs the function
```

**`await` = run this async function.** That's it. You must use `await` to
call async functions, just like you use `()` to call regular functions.

```python
# Regular world:
result = get_user()         # call with ()

# Async world:
result = await get_user()   # call with await
```

### You can have normal sync code inside async functions

Most code inside an `async def` is regular Python. You only need `await`
when calling something that's itself `async def`:

```python
async def process_user():
    response = await fetch_user()   # async — needs await

    name = response["name"]         # regular Python
    upper = name.upper()            # regular Python
    parts = upper.split(" ")        # regular Python

    await save_to_db(result)        # async — needs await
    return result
```

Using `await` on a regular (non-async) value is an error:
```python
name = await response["name"]   # TypeError: object is not awaitable
```

### Why do library methods need `await`?

Functions like `save_to_db()` or `client.get()` aren't ones you wrote —
they come from libraries. But the library author made them `async def`,
so you must `await` them. And because your function uses `await`, it must
be `async def` too. It's a chain:

**Library wrote `async def`** → **you must `await` it** → **your function
must be `async def`** (because only async functions can use `await`).

Under the hood:

```python
# Inside the httpx library, someone wrote:
class Client:
    async def get(self, url):     # <-- they made it async
        ...

# So when you call it, you must await:
response = await client.get(url)
```

If a library uses regular `def`, no await needed:
```python
import requests                        # sync library
response = requests.get(url)           # no await

import httpx                           # async library
response = await client.get(url)       # await required
```

---

## Part 4: Running multiple things concurrently

`await` runs ONE thing and waits for it. To overlap multiple things,
you need `gather` or `create_task`:

```python
# Sequential — 5 seconds total
await asyncio.sleep(3)
await asyncio.sleep(2)

# Concurrent — 3 seconds total (waits overlap)
await asyncio.gather(asyncio.sleep(3), asyncio.sleep(2))
```

### How switching works

Only one coroutine runs at a time. Every `await` is a **switch point** —
Python can pause this coroutine and resume another one.

```python
await asyncio.gather(make_coffee(), make_toast())
```

```
make_coffee: print("start brewing")    ← runs
make_coffee: await sleep(3)            ← pauses, python checks: anyone else?
make_toast:  print("start toasting")   ← runs (coffee is sleeping)
make_toast:  await sleep(2)            ← pauses, both sleeping now
             ... 2 seconds pass ...
make_toast:  print("toast ready")      ← wakes up, runs
             ... 1 more second ...
make_coffee: print("coffee ready")     ← wakes up, runs
```

The prints happen one at a time, in order. The **waiting** overlaps.

### What about pure computation?

If an async function does heavy computation between `await`s, it blocks
everything else during that time. There's nowhere to switch.

```python
async def crunch():
    data = await load_data()      # ← switch point
    result = heavy_math(data)     # 10 seconds, NO switching possible
    await save(result)            # ← switch point

await asyncio.gather(crunch(), crunch())
# The heavy_math parts run one after the other (~20s)
# Only the load/save waits overlap
```

Making `heavy_math` an `async def` and awaiting it doesn't help — if there's
no I/O inside, there's no place to pause.

### The event loop

`asyncio.run(main())` starts the **event loop** — the scheduler that manages
who's waiting and who's ready to run. You rarely interact with it directly.

---

## Part 5: asyncio.Queue — the bridge between producer and consumer

A queue lets one coroutine put stuff in, and another take stuff out.
`asyncio.Queue` is async-aware — the consumer can `await` the next item
while the producer keeps working.

```python
import asyncio

async def producer(queue):
    for i in range(5):
        await asyncio.sleep(0.5)      # simulate doing work
        queue.put_nowait(f"event {i}") # push result into queue (instant)

    queue.put_nowait(None)             # signal: "I'm done"

async def consumer(queue):
    while True:
        event = await queue.get()      # wait for next item
        if event is None:              # done signal
            break
        print(f"  consumed: {event}")

async def main():
    queue = asyncio.Queue()
    await asyncio.gather(producer(queue), consumer(queue))

asyncio.run(main())
```

### Why `put_nowait` vs `await queue.put()`?

- `put_nowait()` — shove it in immediately, never waits. (Our queue has no
  max size, so it can always accept more.)
- `await queue.put()` — waits if the queue is full. We don't need this.
- `await queue.get()` — waits if the queue is EMPTY. This is what the
  consumer uses. It pauses until something appears — and while it's paused,
  the producer can run and push more stuff.

### What happens when the consumer is slower than the producer?

Events buffer in the queue. Nobody gets dropped. The queue just grows.
(See `demo_slow_consumer` in `event_stream.py` — watch the queue size.)

---

## Part 6: asyncio.Future — a one-shot result

A Future is a box that starts empty and gets filled exactly once.

```python
future = asyncio.get_running_loop().create_future()

# Somewhere later:
future.set_result("the answer")    # fills the box (can only do this once)

# Someone waiting:
result = await future              # waits until the box is filled
```

We use this for `stream.result()` — the consumer can await the final result
of the entire stream (all the messages the agent produced).

---

## Part 7: `async for` — iterating over async stuff

If a class defines `__aiter__`, you can use `async for`:

```python
class Countdown:
    def __init__(self, n):
        self.n = n

    async def __aiter__(self):
        for i in range(self.n, 0, -1):
            await asyncio.sleep(0.5)
            yield i

async def main():
    async for num in Countdown(3):
        print(num)
    # prints: 3, 2, 1 (with 0.5s gaps)

asyncio.run(main())
```

`yield` inside an `async def` makes it an **async generator**. Each `yield`
pauses and gives a value to whoever is iterating.

### Why `async for` instead of just `for` with `await` inside?

Because the **iteration itself** needs to await — getting the next value might
require waiting (for a network response, a queue item, etc.):

```python
# Regular for: getting the next item is instant (it's already in memory)
for item in [1, 2, 3]:
    await do_something(item)    # body can await, but next() is instant

# Async for: getting the next item requires waiting
async for event in stream:     # __anext__() awaits — waits for the next event to arrive
    print(event)               # body runs once the item arrives

# What async for is actually doing (syntactic sugar for):
while True:
    try:
        event = await stream.__anext__()  # wait for next item to arrive
    except StopAsyncIteration:
        break
    print(event)
```

`async for` is essentially the consumer pattern in nice syntax. It sits there
waiting for the producer to push the next item, processes it, waits again.

On its own, `async for` behaves just like a regular loop — one iteration
after another, in order. The async part only matters when other coroutines
are running concurrently:

```python
import asyncio

# async generator function — yield + async def = Python adds __aiter__ for you
# (our EventStream uses a class with explicit __aiter__ because it needs
# to hold state like the queue and future, but the concept is the same)
async def countdown():
    for i in [3, 2, 1]:
        await asyncio.sleep(0.5)   # switch point — other coroutines can run here
        yield i                     # give value to whoever is doing `async for`

async def heartbeat():
    for _ in range(5):
        await asyncio.sleep(0.3)   # switch point
        print("  heartbeat")

async def main():
    async def run_countdown():
        async for num in countdown():   # each iteration awaits the next yield
            print(f"  countdown: {num}")

    # gather runs both concurrently — heartbeat runs during countdown's awaits
    await asyncio.gather(run_countdown(), heartbeat())

asyncio.run(main())
# Output (interleaved — heartbeat runs during countdown's awaits):
# heartbeat
# countdown: 3
# heartbeat
# heartbeat
# countdown: 2
# heartbeat
# countdown: 1
# heartbeat
```

Without `gather`, the countdown would just print 3, 2, 1 with no
heartbeat interleaved.

---

## Part 8: Our EventStream — putting it all together

Now we have all the pieces:

- **asyncio.Queue** → buffer events between producer and consumer
- **asyncio.Future** → hold the final result
- **`async for` / `__aiter__`** → let consumers iterate naturally
- **Sentinel value** → signal "stream is done" through the queue

See `event_stream.py` in this folder for the implementation + experiments.

---

## Part 9: How this fits into the agent

```
agent_loop (producer)              consumer (UI, CLI, etc.)
    │                                   │
    ├─ push(agent_start)          ──►   async for event in stream:
    ├─ push(message_update)       ──►       print(event)
    ├─ push(message_update)       ──►       print(event)
    ├─ push(tool_execution_start) ──►       print(event)
    ├─ push(tool_execution_end)   ──►       print(event)
    ├─ push(message_end)          ──►       print(event)
    └─ end(all_messages)          ──►   # loop exits
                                        result = await stream.result()
```

The agent loop runs as a background task (via `asyncio.create_task`).
The consumer iterates the stream. They run concurrently — the producer
pushes events whenever it has them, and the consumer processes them
whenever it's ready. The queue bridges the gap.

