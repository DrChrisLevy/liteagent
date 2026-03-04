import asyncio
import pytest
from liteagent.stream import EventStream


# --- Basic push/iterate/end ---


async def test_push_and_iterate():
    stream = EventStream()
    stream.push("a")
    stream.push("b")
    stream.end()

    events = [e async for e in stream]
    assert events == ["a", "b"]


async def test_end_stores_result():
    stream = EventStream()
    stream.push("x")
    stream.end({"answer": 42})

    assert await stream.result() == {"answer": 42}


async def test_result_default_none():
    stream = EventStream()
    stream.end()

    assert await stream.result() is None


# --- Producer-consumer concurrency ---


async def test_concurrent_producer_consumer():
    stream = EventStream()
    collected = []

    async def producer():
        for i in range(5):
            await asyncio.sleep(0.01)
            stream.push(i)
        stream.end("done")

    async def consumer():
        async for event in stream:
            collected.append(event)

    await asyncio.gather(producer(), consumer())
    assert collected == [0, 1, 2, 3, 4]
    assert await stream.result() == "done"


# --- Edge cases ---


async def test_push_after_end_ignored():
    stream = EventStream()
    stream.push("before")
    stream.end("result")
    stream.push("after")  # should be ignored

    events = [e async for e in stream]
    assert events == ["before"]


async def test_double_end_ignored():
    stream = EventStream()
    stream.end("first")
    stream.end("second")  # should be ignored

    assert await stream.result() == "first"


async def test_none_is_valid_event():
    stream = EventStream()
    stream.push(None)
    stream.push("real")
    stream.end()

    events = [e async for e in stream]
    assert events == [None, "real"]


async def test_empty_stream():
    stream = EventStream()
    stream.end("empty")

    events = [e async for e in stream]
    assert events == []
    assert await stream.result() == "empty"


# --- Result awaiting ---


async def test_result_blocks_until_end():
    stream = EventStream()

    async def delayed_end():
        await asyncio.sleep(0.05)
        stream.end("delayed")

    asyncio.create_task(delayed_end())
    result = await stream.result()
    assert result == "delayed"


# --- Event types ---


@pytest.mark.parametrize(
    "event",
    [
        {"type": "agent_start"},
        {"type": "message_update", "delta": "hello"},
        [1, 2, 3],
        "plain string",
        42,
    ],
)
async def test_any_type_can_be_pushed(event):
    stream = EventStream()
    stream.push(event)
    stream.end()

    events = [e async for e in stream]
    assert events == [event]
