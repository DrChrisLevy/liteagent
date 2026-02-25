import asyncio

_SENTINEL = object()


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
