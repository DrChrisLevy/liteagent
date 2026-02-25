"""
EventStream — async producer-consumer queue.

This is the foundation of py-pi-agent. The agent loop pushes events,
consumers iterate over them with `async for`.

Play with this in ipython:
    $ uv run ipython
    >>> import asyncio
    >>> from learnings.event_stream.event_stream import *
    >>> asyncio.run(demo_basic())
    >>> asyncio.run(demo_fast_producer())
    >>> asyncio.run(demo_result())
"""

import asyncio

# Sentinel object — used to signal "stream is done" through the queue.
# We use a unique object (not None) so that None can be a valid event.
_SENTINEL = object()


class EventStream:
    """Async event stream with producer-consumer pattern.

    Producer side (the agent loop):
        stream.push(event)   — non-blocking, queues an event
        stream.end(result)   — signals completion, stores final result

    Consumer side (UI, CLI, SSE, whatever):
        async for event in stream:  — iterates events as they arrive
        await stream.result()       — gets the final result after stream ends
    """

    def __init__(self):
        self._queue = asyncio.Queue()
        self._done = False
        # Future that holds the final result (filled when end() is called)
        self._result_future = asyncio.get_running_loop().create_future()

    def push(self, event):
        """Push an event to the stream. Non-blocking.

        If a consumer is waiting (blocked on `async for`), it gets the event
        immediately. Otherwise the event sits in the queue until the consumer
        asks for it.

        Does nothing if the stream is already ended.
        """
        if self._done:
            return
        self._queue.put_nowait(event)

    def end(self, result=None):
        """Signal that the stream is complete. Stores the final result.

        After this, push() does nothing and the consumer's `async for` exits.
        """
        if self._done:
            return
        self._done = True
        if not self._result_future.done():
            self._result_future.set_result(result)
        # Push sentinel so the consumer's async for loop knows to stop
        self._queue.put_nowait(_SENTINEL)

    async def result(self):
        """Await the final result. Blocks until end() is called."""
        return await self._result_future

    async def __aiter__(self):
        """Iterate over events until the stream ends.

        Usage:
            async for event in stream:
                print(event)
        """
        while True:
            event = await self._queue.get()
            if event is _SENTINEL:
                return
            yield event


# ============================================================================
# Experiments — run these in ipython to build intuition
# ============================================================================


async def demo_basic():
    """Most basic demo: push some events, iterate them.

    >>> asyncio.run(demo_basic())
    """
    stream = EventStream()

    # Push events first, THEN iterate (events buffer in queue)
    stream.push({"type": "hello", "text": "first event"})
    stream.push({"type": "hello", "text": "second event"})
    stream.end("done!")

    # Now consume — all 3 events are already in the queue
    async for event in stream:
        print(f"got: {event}")

    # Get the final result
    result = await stream.result()
    print(f"result: {result}")


async def demo_fast_producer():
    """Producer and consumer running at the same time.

    The producer pushes events every 0.3s.
    The consumer processes each one (takes 0.1s).
    Events flow through in real time.

    >>> asyncio.run(demo_fast_producer())
    """
    stream = EventStream()

    async def producer():
        for i in range(5):
            await asyncio.sleep(0.3)
            event = {"type": "update", "n": i}
            stream.push(event)
            print(f"  [producer] pushed {i}")
        stream.end({"total_events": 5})
        print("  [producer] ended stream")

    async def consumer():
        async for event in stream:
            print(f"  [consumer] got {event}")
            await asyncio.sleep(0.1)  # simulate processing
        print(f"  [consumer] stream ended, result = {await stream.result()}")

    # Run both concurrently — this is the key!
    await asyncio.gather(producer(), consumer())


async def demo_slow_consumer():
    """What happens when the consumer is slower than the producer?

    Producer pushes every 0.1s, consumer takes 0.5s per event.
    Events buffer in the queue — nobody gets dropped.

    >>> asyncio.run(demo_slow_consumer())
    """
    stream = EventStream()

    async def producer():
        for i in range(5):
            await asyncio.sleep(0.1)
            stream.push({"n": i})
            print(f"  [producer] pushed {i} (queue size: {stream._queue.qsize()})")
        stream.end("all done")

    async def consumer():
        async for event in stream:
            print(f"  [consumer] processing {event}...")
            await asyncio.sleep(0.5)  # slow!
        print("  [consumer] done")

    await asyncio.gather(producer(), consumer())


async def demo_result():
    """You can await the result without iterating events.

    Maybe you don't care about the streaming events — you just want the
    final answer. result() waits until end() is called.

    >>> asyncio.run(demo_result())
    """
    stream = EventStream()

    async def producer():
        print("(producer started)")
        stream.push({"type": "working"})
        await asyncio.sleep(1)
        stream.push({"type": "still working"})
        await asyncio.sleep(1)
        stream.end(["message1", "message2"])  # the final result

    asyncio.create_task(producer())  # schedules producer, doesn't run it yet
    print("(task scheduled, producer hasn't run yet)")

    # The await below is the first switch point — producer starts running here
    result = await stream.result()
    print(f"final result: {result}")


async def demo_like_agent():
    """Simulates how the real agent loop will use this.

    The "agent loop" runs as a background task, pushing events.
    The consumer iterates and prints them — like a CLI would.

    >>> asyncio.run(demo_like_agent())
    """
    stream = EventStream()
    all_messages = []

    async def fake_agent_loop():
        """Pretend to be the agent loop."""
        stream.push({"type": "agent_start"})

        # Pretend: user message
        user_msg = {"role": "user", "content": "hello"}
        all_messages.append(user_msg)
        stream.push({"type": "message_start", "message": user_msg})
        stream.push({"type": "message_end", "message": user_msg})

        # Pretend: LLM streams a response
        assistant_msg = {"role": "assistant", "content": ""}
        stream.push({"type": "message_start", "message": assistant_msg})

        for word in ["Hello", " there", "!", " How", " can", " I", " help", "?"]:
            await asyncio.sleep(0.15)  # simulate streaming delay
            assistant_msg["content"] += word
            stream.push(
                {
                    "type": "message_update",
                    "delta_type": "text_delta",
                    "delta": word,
                }
            )

        all_messages.append(assistant_msg)
        stream.push({"type": "message_end", "message": assistant_msg})
        stream.push({"type": "agent_end", "messages": all_messages})
        stream.end(all_messages)

    asyncio.create_task(fake_agent_loop())  # schedules loop, doesn't run it yet
    print("(task scheduled, producer hasn't run yet)")

    # Consume events — just like a real CLI would
    # The first `await queue.get()` inside __aiter__ is where the producer starts running
    async for event in stream:
        if event["type"] == "message_update" and event["delta_type"] == "text_delta":
            print(event["delta"], end="", flush=True)
        elif event["type"] == "agent_end":
            print()  # newline after streaming
            print(f"\n--- agent done, {len(event['messages'])} messages ---")


# Quick way to run all demos
async def run_all():
    demos = [
        ("basic", demo_basic),
        ("fast producer", demo_fast_producer),
        ("slow consumer", demo_slow_consumer),
        ("result only", demo_result),
        ("like agent", demo_like_agent),
    ]
    for name, fn in demos:
        print(f"\n{'=' * 50}")
        print(f"  {name}")
        print(f"{'=' * 50}")
        await fn()


if __name__ == "__main__":
    asyncio.run(run_all())
