"""Shared mock helpers for litellm in tests."""

import json

from liteagent.types import Tool, ToolResult


class _Obj:
    """Attribute bag — only has the attrs you set."""

    def __init__(self, **kw):
        self.__dict__.update(kw)


def make_delta(**kw):
    return _Obj(**{k: v for k, v in kw.items() if v is not None})


def make_chunk(delta=None, finish_reason=None):
    return _Obj(
        choices=[_Obj(delta=delta or make_delta(), finish_reason=finish_reason)]
    )


def make_tc_delta(index, id=None, name=None, arguments=None):
    func = _Obj(name=name, arguments=arguments) if (name or arguments) else None
    return _Obj(index=index, id=id, function=func)


def make_final(content=None, tool_calls_raw=None, finish_reason="stop", usage=None):
    tc = None
    if tool_calls_raw:
        tc = [
            _Obj(
                id=t["id"],
                function=_Obj(
                    name=t["function"]["name"], arguments=t["function"]["arguments"]
                ),
            )
            for t in tool_calls_raw
        ]
    msg = _Obj(content=content, tool_calls=tc)
    return _Obj(choices=[_Obj(message=msg, finish_reason=finish_reason)], usage=usage)


async def async_iter(items):
    for item in items:
        yield item


def simple_tool(name="test_tool", execute_fn=None):
    """Build a minimal Tool for testing."""

    async def _default(tool_call_id, params, signal=None, on_update=None):
        return ToolResult(content=[{"type": "text", "text": "ok"}])

    return Tool(
        name=name,
        description=f"Test tool: {name}",
        parameters={"type": "object", "properties": {}},
        execute=execute_fn or _default,
    )


def tc_msg(calls):
    """Build assistant message with tool_calls."""
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": f"call_{i}",
                "function": {
                    "name": name,
                    "arguments": json.dumps(args) if isinstance(args, dict) else args,
                },
            }
            for i, (name, args) in enumerate(calls)
        ],
    }
