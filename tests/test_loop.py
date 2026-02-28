"""
Tests for the core loop (loop.py).

Unit tests run by default. Slow tests (real API calls) need: ./dev test -m slow
Multimodal spike detection runs against all 5 target models.
"""

import asyncio
import base64
import io
import json

import numpy as np
import pytest
from pydantic import BaseModel

from py_pi_agent.loop import (
    _build_tool_result_message,
    _extract_usage,
    _maybe_await,
    _skip_tool_call,
    _validate_tool_args,
    agent_loop,
    agent_loop_continue,
    execute_tool_calls,
    run_loop,
    stream_llm_response,
)
from py_pi_agent.stream import EventStream
from py_pi_agent.types import AgentConfig, AgentContext, Tool, ToolResult

# ── Target models ──────────────────────────────────────────────────────────

ALL_MODELS = [
    "anthropic/claude-sonnet-4-6",
    "anthropic/claude-opus-4-6",
    "gemini/gemini-3-flash-preview",
    "gemini/gemini-3.1-pro-preview",
    "gpt-5.2",
]


# ── convert_to_llm (provider-aware) ───────────────────────────────────────


def make_convert_to_llm(model: str):
    """Build a convert_to_llm that handles OpenAI's string-only tool results."""
    is_openai = model.startswith("gpt") or model.startswith("openai/")

    def convert(messages: list) -> list:
        result = []
        pending_images = []

        for m in messages:
            # Flush pending images before any non-tool message
            if pending_images and m.get("role") != "tool":
                result.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Attached image(s) from tool result:",
                            },
                            *pending_images,
                        ],
                    }
                )
                pending_images = []

            role = m.get("role")

            if role == "assistant":
                msg = {"role": "assistant"}
                if m.get("content"):
                    msg["content"] = m["content"]
                if m.get("tool_calls"):
                    msg["tool_calls"] = m["tool_calls"]
                if m.get("thinking_blocks"):
                    msg["thinking_blocks"] = m["thinking_blocks"]
                if m.get("reasoning_content"):
                    msg["reasoning_content"] = m["reasoning_content"]
                result.append(msg)

            elif role == "user":
                result.append({"role": "user", "content": m["content"]})

            elif role == "tool":
                content = m.get("content")
                if isinstance(content, list):
                    text_parts = [b["text"] for b in content if b.get("type") == "text"]
                    images = [b for b in content if b.get("type") == "image_url"]
                    text = "\n".join(text_parts)

                    if is_openai and images:
                        result.append(
                            {
                                "role": "tool",
                                "tool_call_id": m["tool_call_id"],
                                "content": text or "(see attached image)",
                            }
                        )
                        pending_images.extend(images)
                    elif images:
                        llm_blocks = [
                            b for b in content if b.get("type") in ("text", "image_url")
                        ]
                        result.append(
                            {
                                "role": "tool",
                                "tool_call_id": m["tool_call_id"],
                                "content": llm_blocks,
                            }
                        )
                    else:
                        result.append(
                            {
                                "role": "tool",
                                "tool_call_id": m["tool_call_id"],
                                "content": text,
                            }
                        )
                else:
                    result.append(
                        {
                            "role": "tool",
                            "tool_call_id": m["tool_call_id"],
                            "content": content,
                        }
                    )

        # Flush remaining images
        if pending_images:
            result.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Attached image(s) from tool result:"},
                        *pending_images,
                    ],
                }
            )

        return result

    return convert


# ── Test tools ─────────────────────────────────────────────────────────────


async def echo_exec(tool_call_id, params, signal=None, on_update=None):
    return ToolResult(content=[{"type": "text", "text": params["message"]}])


ECHO_TOOL = Tool(
    name="echo",
    description="Echo back a message",
    parameters={
        "type": "object",
        "properties": {
            "message": {"type": "string", "description": "The message to echo"}
        },
        "required": ["message"],
    },
    execute=echo_exec,
)


async def generate_chart_exec(tool_call_id, params, signal=None, on_update=None):
    """Generate a simple bar chart with one obvious outlier. Returns text + image."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    title = params.get("title", "Monthly Server Errors")

    # 6 months, flat baseline ~50, one random month spikes to 500+
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun"]
    values = [50, 48, 52, 47, 51, 49]
    spike_idx = np.random.randint(0, 6)
    values[spike_idx] = np.random.randint(400, 600)
    spike_month = months[spike_idx]

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ["#e74c3c" if i == spike_idx else "#3498db" for i in range(6)]
    ax.bar(months, values, color=colors)
    ax.set_title(title)
    ax.set_ylabel("Errors")
    for i, v in enumerate(values):
        ax.text(i, v + 10, str(v), ha="center", fontweight="bold", fontsize=12)
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    img_b64 = base64.b64encode(buf.getvalue()).decode()

    return ToolResult(
        content=[
            {"type": "text", "text": f"Generated chart: {title}"},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{img_b64}"},
            },
        ],
        details={"spike_month": spike_month, "spike_idx": spike_idx},
    )


CHART_TOOL = Tool(
    name="generate_chart",
    description="Generate a time series chart of server response times. Returns an image. The data has anomalies you can analyze.",
    parameters={
        "type": "object",
        "properties": {
            "days": {
                "type": "integer",
                "description": "Number of days to plot (default 90)",
            },
            "title": {"type": "string", "description": "Chart title"},
        },
    },
    execute=generate_chart_exec,
)


# ── Helpers ────────────────────────────────────────────────────────────────


async def collect_events(stream: EventStream) -> list[dict]:
    return [event async for event in stream]


def event_types(events: list[dict]) -> list[str]:
    return [e["type"] for e in events]


# ── Mock infrastructure ────────────────────────────────────────────────────


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


def make_context(system_prompt="You are helpful.", messages=None, tools=None):
    return AgentContext(
        system_prompt=system_prompt,
        messages=messages if messages is not None else [],
        tools=tools,
    )


def make_config(model="test-model", **kw):
    return AgentConfig(
        model=model,
        convert_to_llm=kw.pop("convert_to_llm", lambda msgs: msgs),
        **kw,
    )


def _simple_tool(name="test_tool", execute_fn=None):
    """Build a minimal Tool for testing."""

    async def _default(tool_call_id, params, signal=None, on_update=None):
        return ToolResult(content=[{"type": "text", "text": "ok"}])

    return Tool(
        name=name,
        description=f"Test tool: {name}",
        parameters={"type": "object", "properties": {}},
        execute=execute_fn or _default,
    )


def _tc_msg(calls):
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


@pytest.fixture
def mock_llm(monkeypatch):
    """Single-turn litellm mock. Returns captured kwargs dict."""
    captured = {}

    def _mock(chunks, final):
        async def fake_acompletion(**kwargs):
            captured.update(kwargs)
            return async_iter(chunks)

        monkeypatch.setattr("py_pi_agent.loop.litellm.acompletion", fake_acompletion)
        monkeypatch.setattr(
            "py_pi_agent.loop.litellm.stream_chunk_builder", lambda _: final
        )
        return captured

    return _mock


@pytest.fixture
def mock_llm_seq(monkeypatch):
    """Multi-turn litellm mock. Takes list of (chunks, final) per LLM call."""

    def _mock(turns):
        remaining_chunks = [t[0] for t in turns]
        remaining_finals = [t[1] for t in turns]

        async def fake_acompletion(**kwargs):
            return async_iter(remaining_chunks.pop(0))

        monkeypatch.setattr("py_pi_agent.loop.litellm.acompletion", fake_acompletion)
        monkeypatch.setattr(
            "py_pi_agent.loop.litellm.stream_chunk_builder",
            lambda _: remaining_finals.pop(0),
        )

    return _mock


# ── Group 1: Pure helpers ─────────────────────────────────────────────────


async def test_maybe_await_sync():
    assert await _maybe_await(lambda x: x + 1, 2) == 3


async def test_maybe_await_async():
    async def add(x):
        return x + 1

    assert await _maybe_await(add, 2) == 3


async def test_extract_usage_none():
    u = _extract_usage(None)
    assert u == {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "cache_read_tokens": 0,
        "cache_creation_tokens": 0,
    }


async def test_extract_usage_with_cache_details():
    details = _Obj(cached_tokens=500, cache_creation_tokens=100)
    usage = _Obj(
        prompt_tokens=1000,
        completion_tokens=200,
        total_tokens=1200,
        prompt_tokens_details=details,
    )
    u = _extract_usage(usage)
    assert u["prompt_tokens"] == 1000
    assert u["cache_read_tokens"] == 500
    assert u["cache_creation_tokens"] == 100


async def test_validate_tool_args_no_model():
    tool = _simple_tool()
    assert _validate_tool_args(tool, '{"x": 1}') == {"x": 1}
    assert _validate_tool_args(tool, {"x": 1}) == {"x": 1}


async def test_validate_tool_args_coercion():
    class Params(BaseModel):
        count: int

    tool = _simple_tool()
    tool.params_model = Params
    result = _validate_tool_args(tool, {"count": "42"})
    assert result == {"count": 42}
    assert isinstance(result["count"], int)


async def test_validate_tool_args_rejects():
    class Params(BaseModel):
        count: int

    tool = _simple_tool()
    tool.params_model = Params
    with pytest.raises(Exception):
        _validate_tool_args(tool, {"count": "not_a_number"})


async def test_build_tool_result_message_shape():
    result = ToolResult(content=[{"type": "text", "text": "hi"}])
    msg = _build_tool_result_message("call_0", "echo", result, False)
    assert msg["role"] == "tool"
    assert msg["tool_call_id"] == "call_0"
    assert msg["name"] == "echo"
    assert msg["content"] == [{"type": "text", "text": "hi"}]
    assert msg["details"] == {}
    assert msg["is_error"] is False
    assert "timestamp" in msg


# ── Group 2: _skip_tool_call ─────────────────────────────────────────────


async def test_skip_tool_call():
    stream = EventStream()
    tc = {"id": "call_0", "function": {"name": "bash", "arguments": '{"cmd": "ls"}'}}
    msg = _skip_tool_call(tc, stream)
    stream.end()
    events = await collect_events(stream)
    types = event_types(events)

    assert types == [
        "tool_execution_start",
        "tool_execution_end",
        "message_start",
        "message_end",
    ]
    assert events[0]["tool_name"] == "bash"
    assert events[0]["args"] == {"cmd": "ls"}
    assert events[1]["is_error"] is True
    assert msg["is_error"] is True
    assert "Skipped" in msg["content"][0]["text"]
    assert msg["details"] == {}


# ── Group 3: execute_tool_calls ───────────────────────────────────────────


async def test_execute_single_tool_success():
    stream = EventStream()
    tool = _simple_tool("echo", echo_exec)
    assistant = _tc_msg([("echo", {"message": "hi"})])
    result = await execute_tool_calls([tool], assistant, None, stream, None)
    stream.end()
    events = await collect_events(stream)

    assert len(result["tool_results"]) == 1
    assert result["tool_results"][0]["is_error"] is False
    assert "hi" in result["tool_results"][0]["content"][0]["text"]
    assert "tool_execution_start" in event_types(events)
    assert "tool_execution_end" in event_types(events)


async def test_execute_tool_not_found():
    stream = EventStream()
    assistant = _tc_msg([("nonexistent", {})])
    result = await execute_tool_calls([], assistant, None, stream, None)
    stream.end()

    assert len(result["tool_results"]) == 1
    assert result["tool_results"][0]["is_error"] is True
    assert "not found" in result["tool_results"][0]["content"][0]["text"].lower()


async def test_execute_tool_raises():
    async def boom(tool_call_id, params, signal=None, on_update=None):
        raise RuntimeError("kaboom")

    stream = EventStream()
    tool = _simple_tool("boom", boom)
    assistant = _tc_msg([("boom", {})])
    result = await execute_tool_calls([tool], assistant, None, stream, None)
    stream.end()

    assert result["tool_results"][0]["is_error"] is True
    assert "kaboom" in result["tool_results"][0]["content"][0]["text"]


async def test_execute_multiple_sequential():
    order = []

    async def track(name):
        async def fn(tool_call_id, params, signal=None, on_update=None):
            order.append(name)
            return ToolResult(content=[{"type": "text", "text": name}])

        return fn

    tools = [
        _simple_tool("a", await track("a")),
        _simple_tool("b", await track("b")),
    ]
    assistant = _tc_msg([("a", {}), ("b", {})])
    stream = EventStream()
    result = await execute_tool_calls(tools, assistant, None, stream, None)
    stream.end()

    assert order == ["a", "b"]
    assert len(result["tool_results"]) == 2


async def test_execute_on_update():
    async def streamer(tool_call_id, params, signal=None, on_update=None):
        if on_update:
            on_update(ToolResult(content=[{"type": "text", "text": "partial"}]))
        return ToolResult(content=[{"type": "text", "text": "done"}])

    stream = EventStream()
    tool = _simple_tool("streamer", streamer)
    assistant = _tc_msg([("streamer", {"x": 1})])
    await execute_tool_calls([tool], assistant, None, stream, None)
    stream.end()
    events = await collect_events(stream)

    updates = [e for e in events if e["type"] == "tool_execution_update"]
    assert len(updates) == 1
    assert updates[0]["tool_name"] == "streamer"
    assert updates[0]["args"] == {"x": 1}
    assert updates[0]["partial"]["content"][0]["text"] == "partial"


async def test_execute_invalid_json_args():
    stream = EventStream()
    tool = _simple_tool("t")
    assistant = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {"id": "call_0", "function": {"name": "t", "arguments": "not json {"}}
        ],
    }
    result = await execute_tool_calls([tool], assistant, None, stream, None)
    stream.end()

    assert result["tool_results"][0]["is_error"] is True


async def test_execute_pydantic_validation_error():
    class Params(BaseModel):
        count: int

    stream = EventStream()
    tool = _simple_tool("t")
    tool.params_model = Params
    assistant = _tc_msg([("t", {"count": "not_a_number"})])
    result = await execute_tool_calls([tool], assistant, None, stream, None)
    stream.end()

    assert result["tool_results"][0]["is_error"] is True
    assert result["steering_messages"] is None


async def test_execute_steering_skips_remaining():
    call_count = 0

    async def counting(tool_call_id, params, signal=None, on_update=None):
        nonlocal call_count
        call_count += 1
        return ToolResult(content=[{"type": "text", "text": "ok"}])

    tools = [
        _simple_tool("a", counting),
        _simple_tool("b", counting),
        _simple_tool("c", counting),
    ]
    assistant = _tc_msg([("a", {}), ("b", {}), ("c", {})])
    stream = EventStream()

    steering_msg = [{"role": "user", "content": "stop!", "timestamp": 0}]
    call_idx = 0

    async def get_steering():
        nonlocal call_idx
        call_idx += 1
        return steering_msg if call_idx == 1 else []

    result = await execute_tool_calls(tools, assistant, None, stream, get_steering)
    stream.end()
    await collect_events(stream)

    assert call_count == 1  # only first tool ran
    assert len(result["tool_results"]) == 3  # all 3 have results
    assert result["tool_results"][0]["is_error"] is False
    assert result["tool_results"][1]["is_error"] is True  # skipped
    assert result["tool_results"][2]["is_error"] is True  # skipped
    assert result["steering_messages"] == steering_msg


async def test_execute_no_steering_hook():
    stream = EventStream()
    tools = [_simple_tool("a"), _simple_tool("b")]
    assistant = _tc_msg([("a", {}), ("b", {})])
    result = await execute_tool_calls(tools, assistant, None, stream, None)
    stream.end()

    assert len(result["tool_results"]) == 2
    assert all(r["is_error"] is False for r in result["tool_results"])
    assert result["steering_messages"] is None


async def test_execute_error_doesnt_block_later():
    async def fail(tool_call_id, params, signal=None, on_update=None):
        raise RuntimeError("fail")

    stream = EventStream()
    tools = [_simple_tool("bad", fail), _simple_tool("good")]
    assistant = _tc_msg([("bad", {}), ("good", {})])
    result = await execute_tool_calls(tools, assistant, None, stream, None)
    stream.end()

    assert result["tool_results"][0]["is_error"] is True
    assert result["tool_results"][1]["is_error"] is False


async def test_execute_details_preserved():
    async def detailed(tool_call_id, params, signal=None, on_update=None):
        return ToolResult(
            content=[{"type": "text", "text": "ok"}],
            details={"key": "val", "num": 42},
        )

    stream = EventStream()
    tool = _simple_tool("d", detailed)
    assistant = _tc_msg([("d", {})])
    result = await execute_tool_calls([tool], assistant, None, stream, None)
    stream.end()

    assert result["tool_results"][0]["details"] == {"key": "val", "num": 42}


# ── Group 4: stream_llm_response (thin litellm mock) ─────────────────────


async def test_stream_message_start_once(mock_llm):
    chunks = [
        make_chunk(make_delta(content="Hello")),
        make_chunk(make_delta(content=" world")),
    ]
    mock_llm(chunks, make_final(content="Hello world"))
    ctx = make_context(messages=[{"role": "user", "content": "hi"}])
    stream = EventStream()
    await stream_llm_response(ctx, make_config(), None, stream)
    stream.end()
    events = await collect_events(stream)

    starts = [e for e in events if e["type"] == "message_start"]
    assert len(starts) == 1


async def test_stream_text_delta_events(mock_llm):
    chunks = [
        make_chunk(make_delta(content="Hello")),
        make_chunk(make_delta(content=" world")),
    ]
    mock_llm(chunks, make_final(content="Hello world"))
    ctx = make_context(messages=[{"role": "user", "content": "hi"}])
    stream = EventStream()
    msg = await stream_llm_response(ctx, make_config(), None, stream)
    stream.end()
    events = await collect_events(stream)

    updates = [e for e in events if e["type"] == "message_update"]
    assert len(updates) == 2
    assert all(e["delta_type"] == "text_delta" for e in updates)
    assert msg["content"] == "Hello world"


async def test_stream_thinking_delta_events(mock_llm):
    chunks = [make_chunk(make_delta(reasoning_content="thinking..."))]
    mock_llm(chunks, make_final(content=None))
    ctx = make_context(messages=[{"role": "user", "content": "hi"}])
    stream = EventStream()
    await stream_llm_response(ctx, make_config(), None, stream)
    stream.end()
    events = await collect_events(stream)

    updates = [e for e in events if e["type"] == "message_update"]
    assert len(updates) == 1
    assert updates[0]["delta_type"] == "thinking_delta"


async def test_stream_tool_call_delta_events(mock_llm):
    chunks = [
        make_chunk(make_delta(tool_calls=[make_tc_delta(0, id="call_0", name="bash")])),
        make_chunk(
            make_delta(tool_calls=[make_tc_delta(0, arguments='{"cmd": "ls"}')])
        ),
    ]
    tc_raw = [
        {"id": "call_0", "function": {"name": "bash", "arguments": '{"cmd": "ls"}'}}
    ]
    mock_llm(chunks, make_final(tool_calls_raw=tc_raw, finish_reason="tool_calls"))
    ctx = make_context(messages=[{"role": "user", "content": "hi"}])
    stream = EventStream()
    msg = await stream_llm_response(ctx, make_config(), None, stream)
    stream.end()
    events = await collect_events(stream)

    updates = [
        e
        for e in events
        if e["type"] == "message_update" and e["delta_type"] == "tool_call_delta"
    ]
    assert len(updates) == 2
    assert msg["tool_calls"][0]["id"] == "call_0"
    assert msg["tool_calls"][0]["function"]["name"] == "bash"


async def test_stream_empty_chunks(mock_llm):
    mock_llm([], make_final())
    ctx = make_context(messages=[{"role": "user", "content": "hi"}])
    stream = EventStream()
    msg = await stream_llm_response(ctx, make_config(), None, stream)
    stream.end()
    events = await collect_events(stream)

    assert msg["content"] is None
    assert msg["stop_reason"] == "stop"
    assert "message_start" in event_types(events)
    assert "message_end" in event_types(events)


async def test_stream_abort_mid_stream(mock_llm):
    signal = asyncio.Event()
    signal.set()
    mock_llm(
        [make_chunk(make_delta(content="partial"))],
        make_final(content="partial"),
    )
    ctx = make_context(messages=[{"role": "user", "content": "hi"}])
    stream = EventStream()
    msg = await stream_llm_response(ctx, make_config(), signal, stream)
    stream.end()

    assert msg["stop_reason"] == "aborted"


async def test_stream_exception(mock_llm, monkeypatch):
    async def explode(**kwargs):
        raise RuntimeError("LLM down")

    monkeypatch.setattr("py_pi_agent.loop.litellm.acompletion", explode)
    ctx = make_context(messages=[{"role": "user", "content": "hi"}])
    stream = EventStream()
    msg = await stream_llm_response(ctx, make_config(), None, stream)
    stream.end()
    events = await collect_events(stream)

    assert msg["stop_reason"] == "error"
    assert "LLM down" in msg["error_message"]
    assert "message_end" in event_types(events)


async def test_stream_transform_context_called(mock_llm):
    transformed = False

    def transform(msgs, signal=None):
        nonlocal transformed
        transformed = True
        return msgs

    mock_llm([make_chunk(make_delta(content="ok"))], make_final(content="ok"))
    ctx = make_context(messages=[{"role": "user", "content": "hi"}])
    stream = EventStream()
    await stream_llm_response(
        ctx, make_config(transform_context=transform), None, stream
    )
    stream.end()

    assert transformed


@pytest.mark.parametrize(
    "system_prompt, expect_system",
    [("You are helpful.", True), ("", False)],
)
async def test_stream_system_prompt(mock_llm, system_prompt, expect_system):
    captured = mock_llm(
        [make_chunk(make_delta(content="ok"))], make_final(content="ok")
    )
    ctx = make_context(system_prompt=system_prompt)
    ctx.messages.append({"role": "user", "content": "hi"})
    stream = EventStream()
    await stream_llm_response(ctx, make_config(), None, stream)
    stream.end()

    msgs = captured["messages"]
    if expect_system:
        assert msgs[0]["role"] == "system"
        assert msgs[0]["content"] == system_prompt
    else:
        assert not any(m.get("role") == "system" for m in msgs)


async def test_stream_no_tools(mock_llm):
    captured = mock_llm(
        [make_chunk(make_delta(content="ok"))], make_final(content="ok")
    )
    ctx = make_context()
    ctx.messages.append({"role": "user", "content": "hi"})
    stream = EventStream()
    await stream_llm_response(ctx, make_config(), None, stream)
    stream.end()

    assert "tools" not in captured


@pytest.mark.parametrize(
    "kwarg, value",
    [
        ("reasoning_effort", "high"),
        ("max_tokens", 1000),
        ("temperature", 0.5),
        ("num_retries", 3),
    ],
)
async def test_stream_kwargs_conditional(mock_llm, kwarg, value):
    captured = mock_llm(
        [make_chunk(make_delta(content="ok"))], make_final(content="ok")
    )
    ctx = make_context(messages=[{"role": "user", "content": "hi"}])
    stream = EventStream()
    await stream_llm_response(ctx, make_config(**{kwarg: value}), None, stream)
    stream.end()

    assert captured[kwarg] == value


# ── Group 5: run_loop (thin litellm mock) ─────────────────────────────────


async def test_run_loop_single_turn(mock_llm):
    mock_llm(
        [make_chunk(make_delta(content="Hello"))],
        make_final(content="Hello"),
    )
    ctx = make_context(messages=[{"role": "user", "content": "hi"}])
    stream = EventStream()
    new_msgs = []
    await run_loop(ctx, new_msgs, make_config(), None, stream)
    events = await collect_events(stream)
    types = event_types(events)

    assert types[-1] == "agent_end"
    assert len(new_msgs) == 1
    assert new_msgs[0]["role"] == "assistant"


async def test_run_loop_tool_round_trip(mock_llm_seq):
    tc_raw = [
        {"id": "call_0", "function": {"name": "echo", "arguments": '{"message":"hi"}'}}
    ]
    mock_llm_seq(
        [
            (
                [
                    make_chunk(
                        make_delta(
                            tool_calls=[make_tc_delta(0, id="call_0", name="echo")]
                        )
                    )
                ],
                make_final(tool_calls_raw=tc_raw, finish_reason="tool_calls"),
            ),
            (
                [make_chunk(make_delta(content="Done"))],
                make_final(content="Done"),
            ),
        ]
    )
    ctx = make_context(
        messages=[{"role": "user", "content": "echo hi"}],
        tools=[_simple_tool("echo", echo_exec)],
    )
    stream = EventStream()
    new_msgs = []
    await run_loop(ctx, new_msgs, make_config(), None, stream)
    events = await collect_events(stream)
    types = event_types(events)

    assert "tool_execution_start" in types
    assert "tool_execution_end" in types
    # assistant (tool call) + tool result + assistant (final)
    assert sum(1 for m in new_msgs if m.get("role") == "assistant") == 2
    assert sum(1 for m in new_msgs if m.get("role") == "tool") == 1


async def test_run_loop_error_exits(mock_llm, monkeypatch):
    async def explode(**kwargs):
        raise RuntimeError("LLM down")

    monkeypatch.setattr("py_pi_agent.loop.litellm.acompletion", explode)
    ctx = make_context(messages=[{"role": "user", "content": "hi"}])
    stream = EventStream()
    new_msgs = []
    await run_loop(ctx, new_msgs, make_config(), None, stream)
    events = await collect_events(stream)
    types = event_types(events)

    assert "turn_end" in types
    assert "agent_end" in types
    assert new_msgs[-1]["stop_reason"] == "error"


async def test_run_loop_aborted_exits(mock_llm):
    signal = asyncio.Event()
    signal.set()
    mock_llm([], make_final())
    ctx = make_context(messages=[{"role": "user", "content": "hi"}])
    stream = EventStream()
    new_msgs = []
    await run_loop(ctx, new_msgs, make_config(), signal, stream)
    events = await collect_events(stream)
    types = event_types(events)

    assert "turn_end" in types
    assert "agent_end" in types
    assert new_msgs[-1]["stop_reason"] == "aborted"


async def test_run_loop_steering_injected(mock_llm_seq):
    """Steering arrives after first LLM call → injected before second call."""
    steering_call = 0

    def get_steering():
        nonlocal steering_call
        steering_call += 1
        # First poll (before first LLM call) returns empty.
        # Second poll (after first LLM response, which has tool_calls) returns steering.
        if steering_call == 2:
            return [{"role": "user", "content": "do this instead", "timestamp": 0}]
        return []

    tc_raw = [{"id": "call_0", "function": {"name": "t", "arguments": "{}"}}]
    mock_llm_seq(
        [
            (
                [make_chunk(make_delta(content=""))],
                make_final(tool_calls_raw=tc_raw, finish_reason="tool_calls"),
            ),
            ([make_chunk(make_delta(content="second"))], make_final(content="second")),
        ]
    )
    ctx = make_context(
        messages=[{"role": "user", "content": "hi"}],
        tools=[_simple_tool("t")],
    )
    stream = EventStream()
    new_msgs = []
    await run_loop(
        ctx, new_msgs, make_config(get_steering_messages=get_steering), None, stream
    )
    events = await collect_events(stream)

    # Steering message was injected
    injected = [
        e
        for e in events
        if e["type"] == "message_end"
        and e["message"].get("content") == "do this instead"
    ]
    assert len(injected) == 1
    assert sum(1 for m in new_msgs if m.get("role") == "assistant") == 2


async def test_run_loop_steering_from_tools_priority(mock_llm_seq):
    """Steering found during tool execution should be used, not a fresh poll."""
    poll_count = 0

    async def get_steering():
        nonlocal poll_count
        poll_count += 1
        return []

    tc_raw = [
        {"id": "call_0", "function": {"name": "a", "arguments": "{}"}},
        {"id": "call_1", "function": {"name": "b", "arguments": "{}"}},
    ]

    async def tool_a(tool_call_id, params, signal=None, on_update=None):
        return ToolResult(content=[{"type": "text", "text": "ok"}])

    # Tool a's steering check will find a message, skipping tool b
    steering_found = False
    original_get_steering = get_steering

    async def steering_on_first_tool():
        nonlocal steering_found
        if not steering_found:
            steering_found = True
            return [{"role": "user", "content": "redirect", "timestamp": 0}]
        return await original_get_steering()

    mock_llm_seq(
        [
            (
                [make_chunk(make_delta(content=""))],
                make_final(tool_calls_raw=tc_raw, finish_reason="tool_calls"),
            ),
            ([make_chunk(make_delta(content="final"))], make_final(content="final")),
        ]
    )
    ctx = make_context(
        messages=[{"role": "user", "content": "hi"}],
        tools=[_simple_tool("a", tool_a), _simple_tool("b")],
    )
    stream = EventStream()
    new_msgs = []
    await run_loop(
        ctx,
        new_msgs,
        make_config(get_steering_messages=steering_on_first_tool),
        None,
        stream,
    )
    events = await collect_events(stream)

    # Steering was used (redirect message present)
    redirects = [
        e
        for e in events
        if e["type"] == "message_end" and e["message"].get("content") == "redirect"
    ]
    assert len(redirects) == 1


async def test_run_loop_follow_ups(mock_llm_seq):
    follow_call = 0

    def get_follow_ups():
        nonlocal follow_call
        follow_call += 1
        if follow_call == 1:
            return [{"role": "user", "content": "also do this", "timestamp": 0}]
        return []

    mock_llm_seq(
        [
            ([make_chunk(make_delta(content="first"))], make_final(content="first")),
            ([make_chunk(make_delta(content="second"))], make_final(content="second")),
        ]
    )
    ctx = make_context(messages=[{"role": "user", "content": "hi"}])
    stream = EventStream()
    new_msgs = []
    await run_loop(
        ctx,
        new_msgs,
        make_config(get_follow_up_messages=get_follow_ups),
        None,
        stream,
    )
    events = await collect_events(stream)

    assert sum(1 for m in new_msgs if m.get("role") == "assistant") == 2
    follow_up_injected = [
        e
        for e in events
        if e["type"] == "message_end" and e["message"].get("content") == "also do this"
    ]
    assert len(follow_up_injected) == 1


async def test_run_loop_no_follow_ups(mock_llm):
    mock_llm([make_chunk(make_delta(content="done"))], make_final(content="done"))
    ctx = make_context(messages=[{"role": "user", "content": "hi"}])
    stream = EventStream()
    new_msgs = []
    await run_loop(
        ctx,
        new_msgs,
        make_config(get_follow_up_messages=lambda: []),
        None,
        stream,
    )
    await collect_events(stream)

    assert sum(1 for m in new_msgs if m.get("role") == "assistant") == 1


async def test_run_loop_first_turn_skip(mock_llm_seq):
    """first_turn=True means run_loop skips turn_start on first iteration."""
    tc_raw = [{"id": "call_0", "function": {"name": "t", "arguments": "{}"}}]
    mock_llm_seq(
        [
            (
                [make_chunk(make_delta(content=""))],
                make_final(tool_calls_raw=tc_raw, finish_reason="tool_calls"),
            ),
            ([make_chunk(make_delta(content="done"))], make_final(content="done")),
        ]
    )
    ctx = make_context(
        messages=[{"role": "user", "content": "hi"}],
        tools=[_simple_tool("t")],
    )
    stream = EventStream()
    await run_loop(ctx, [], make_config(), None, stream)
    events = await collect_events(stream)
    types = event_types(events)

    # First event should NOT be turn_start (entry point already emitted it)
    assert types[0] != "turn_start"
    # But there should be a turn_start later (for the second turn)
    assert "turn_start" in types


async def test_run_loop_new_messages_only(mock_llm):
    mock_llm(
        [make_chunk(make_delta(content="reply"))],
        make_final(content="reply"),
    )
    pre_existing = [
        {"role": "user", "content": "old msg", "timestamp": 0},
        {"role": "assistant", "content": "old reply", "timestamp": 1},
        {"role": "user", "content": "new msg", "timestamp": 2},
    ]
    ctx = make_context(messages=list(pre_existing))
    new_msgs = []
    stream = EventStream()
    await run_loop(ctx, new_msgs, make_config(), None, stream)
    await collect_events(stream)

    # new_messages should only have the assistant reply from THIS run
    assert len(new_msgs) == 1
    assert new_msgs[0]["role"] == "assistant"


# ── Group 6: Entry points ────────────────────────────────────────────────


async def test_agent_loop_shallow_copy(mock_llm):
    mock_llm([make_chunk(make_delta(content="hi"))], make_final(content="hi"))
    original_msgs = [{"role": "system", "content": "old"}]
    original_copy = list(original_msgs)
    ctx = make_context(messages=original_msgs)
    user_msg = {"role": "user", "content": "hello", "timestamp": 0}
    stream = agent_loop([user_msg], ctx, make_config())
    await collect_events(stream)

    # Original list should not have been mutated
    assert original_msgs == original_copy


async def test_agent_loop_multiple_prompts(mock_llm):
    mock_llm([make_chunk(make_delta(content="ok"))], make_final(content="ok"))
    ctx = make_context()
    prompts = [
        {"role": "user", "content": "msg1", "timestamp": 0},
        {"role": "user", "content": "msg2", "timestamp": 1},
    ]
    stream = agent_loop(prompts, ctx, make_config())
    events = await collect_events(stream)

    prompt_starts = [
        e
        for e in events
        if e["type"] == "message_start"
        and e["message"].get("content") in ("msg1", "msg2")
    ]
    assert len(prompt_starts) == 2


async def test_agent_loop_result(mock_llm):
    mock_llm([make_chunk(make_delta(content="hello"))], make_final(content="hello"))
    ctx = make_context()
    user_msg = {"role": "user", "content": "hi", "timestamp": 0}
    stream = agent_loop([user_msg], ctx, make_config())
    result = await stream.result()

    assert isinstance(result, list)
    assert any(m.get("role") == "assistant" for m in result)


async def test_agent_loop_safety_net(monkeypatch):
    async def explode(*args, **kwargs):
        raise TypeError("internal bug")

    monkeypatch.setattr("py_pi_agent.loop.run_loop", explode)
    ctx = make_context()
    user_msg = {"role": "user", "content": "hi", "timestamp": 0}
    stream = agent_loop([user_msg], ctx, make_config())
    events = await collect_events(stream)
    types = event_types(events)

    assert "agent_end" in types
    result = await stream.result()
    assert result is not None
    assert result[0]["stop_reason"] == "error"
    assert "internal bug" in result[0]["error_message"]


async def test_agent_loop_continue_from_tool(mock_llm):
    mock_llm(
        [make_chunk(make_delta(content="continued"))],
        make_final(content="continued"),
    )
    ctx = make_context(
        messages=[
            {"role": "user", "content": "hi", "timestamp": 0},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "c0", "function": {"name": "t", "arguments": "{}"}}
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "c0",
                "content": "result",
            },
        ]
    )
    stream = agent_loop_continue(ctx, make_config())
    events = await collect_events(stream)
    types = event_types(events)

    assert types[0] == "agent_start"
    assert types[1] == "turn_start"
    assert "agent_end" in types


async def test_agent_loop_continue_events(mock_llm):
    mock_llm([make_chunk(make_delta(content="ok"))], make_final(content="ok"))
    ctx = make_context(messages=[{"role": "user", "content": "hi", "timestamp": 0}])
    stream = agent_loop_continue(ctx, make_config())
    events = await collect_events(stream)
    types = event_types(events)

    assert types[0] == "agent_start"
    assert types[1] == "turn_start"


@pytest.mark.parametrize("hook_style", ["sync", "async"])
async def test_hooks_sync_and_async(mock_llm, hook_style):
    if hook_style == "sync":
        convert = lambda msgs: msgs  # noqa: E731
    else:

        async def convert(msgs):
            return msgs

    mock_llm([make_chunk(make_delta(content="ok"))], make_final(content="ok"))
    ctx = make_context(messages=[{"role": "user", "content": "hi"}])
    stream = EventStream()
    await stream_llm_response(ctx, make_config(convert_to_llm=convert), None, stream)
    stream.end()
    events = await collect_events(stream)

    assert "message_end" in event_types(events)


async def test_stream_chunk_no_choices(mock_llm):
    """Chunk with no choices (e.g. usage-only) is skipped without crash."""
    chunks = [
        _Obj(choices=[]),  # no choices
        make_chunk(make_delta(content="ok")),
    ]
    mock_llm(chunks, make_final(content="ok"))
    ctx = make_context(messages=[{"role": "user", "content": "hi"}])
    stream = EventStream()
    msg = await stream_llm_response(ctx, make_config(), None, stream)
    stream.end()
    events = await collect_events(stream)

    assert msg["content"] == "ok"
    updates = [e for e in events if e["type"] == "message_update"]
    assert len(updates) == 1


async def test_stream_thinking_blocks_without_reasoning_content(mock_llm):
    """thinking_blocks present but no reasoning_content → thinking_delta emitted."""
    tb = [{"type": "thinking", "thinking": "hmm", "signature": ""}]
    chunks = [make_chunk(make_delta(thinking_blocks=tb))]
    mock_llm(chunks, make_final(content=None))
    ctx = make_context(messages=[{"role": "user", "content": "hi"}])
    stream = EventStream()
    await stream_llm_response(ctx, make_config(), None, stream)
    stream.end()
    events = await collect_events(stream)

    updates = [e for e in events if e["type"] == "message_update"]
    assert len(updates) == 1
    assert updates[0]["delta_type"] == "thinking_delta"


async def test_stream_thinking_blocks_with_reasoning_content_no_double(mock_llm):
    """Both reasoning_content and thinking_blocks on same chunk → only one thinking_delta."""
    tb = [{"type": "thinking", "thinking": "hmm", "signature": ""}]
    chunks = [
        make_chunk(make_delta(reasoning_content="thinking...", thinking_blocks=tb))
    ]
    mock_llm(chunks, make_final(content=None))
    ctx = make_context(messages=[{"role": "user", "content": "hi"}])
    stream = EventStream()
    await stream_llm_response(ctx, make_config(), None, stream)
    stream.end()
    events = await collect_events(stream)

    updates = [e for e in events if e["type"] == "message_update"]
    assert len(updates) == 1  # only one, not two
    assert updates[0]["delta_type"] == "thinking_delta"


async def test_agent_loop_continue_safety_net(monkeypatch):
    """Unhandled exception in agent_loop_continue's _run → stream still ends."""

    async def explode(*args, **kwargs):
        raise TypeError("internal bug")

    monkeypatch.setattr("py_pi_agent.loop.run_loop", explode)
    ctx = make_context(messages=[{"role": "user", "content": "hi", "timestamp": 0}])
    stream = agent_loop_continue(ctx, make_config())
    events = await collect_events(stream)
    types = event_types(events)

    assert "agent_end" in types
    result = await stream.result()
    assert result is not None
    assert result[0]["stop_reason"] == "error"
    assert "internal bug" in result[0]["error_message"]


# ── Unit tests (no API calls) ─────────────────────────────────────────────


async def test_abort_before_streaming():
    """Setting signal before streaming → stop_reason='aborted'."""
    signal = asyncio.Event()
    signal.set()  # abort immediately

    context = AgentContext(
        system_prompt="You are helpful.",
        messages=[],
    )
    config = AgentConfig(
        model="anthropic/claude-sonnet-4-6",
        convert_to_llm=lambda msgs: msgs,
    )

    user_msg = {"role": "user", "content": "hello", "timestamp": 0}
    stream = agent_loop([user_msg], context, config, signal)
    events = await collect_events(stream)
    types = event_types(events)

    assert "agent_start" in types
    assert "agent_end" in types

    # Find the assistant message
    message_ends = [e for e in events if e["type"] == "message_end"]
    assert len(message_ends) >= 1
    assistant = message_ends[-1]["message"]
    assert assistant["stop_reason"] in ("aborted", "error")


async def test_agent_loop_continue_rejects_empty():
    """agent_loop_continue with no messages → ValueError."""
    context = AgentContext(system_prompt="", messages=[])
    config = AgentConfig(model="x", convert_to_llm=lambda m: m)
    with pytest.raises(ValueError, match="no messages"):
        agent_loop_continue(context, config)


async def test_agent_loop_continue_rejects_assistant_last():
    """agent_loop_continue with assistant as last message → ValueError."""
    context = AgentContext(
        system_prompt="",
        messages=[{"role": "assistant", "content": "hi"}],
    )
    config = AgentConfig(model="x", convert_to_llm=lambda m: m)
    with pytest.raises(ValueError, match="assistant"):
        agent_loop_continue(context, config)


# ── Slow tests (real API calls) ───────────────────────────────────────────


@pytest.mark.slow
@pytest.mark.parametrize("model", ALL_MODELS)
async def test_simple_text_response(model):
    """Basic text response — no tools. Verify event sequence and message shape."""
    context = AgentContext(
        system_prompt="Reply with exactly one sentence.",
        messages=[],
    )
    config = AgentConfig(model=model, convert_to_llm=make_convert_to_llm(model))

    user_msg = {"role": "user", "content": "What is 2+2?", "timestamp": 0}
    stream = agent_loop([user_msg], context, config)
    events = await collect_events(stream)
    types = event_types(events)

    # Event sequence
    assert types[0] == "agent_start"
    assert types[1] == "turn_start"
    assert types[2] == "message_start"  # user prompt
    assert types[3] == "message_end"  # user prompt
    assert "message_start" in types[4:]  # assistant message_start
    assert types[-1] == "agent_end"

    # Find assistant message_end
    assistant_ends = [
        e
        for e in events
        if e["type"] == "message_end" and e["message"].get("role") == "assistant"
    ]
    assert len(assistant_ends) == 1
    assistant = assistant_ends[0]["message"]

    assert assistant["content"]  # has text
    assert assistant["stop_reason"] == "stop"
    assert assistant["usage"]["total_tokens"] > 0

    # Streaming updates happened
    updates = [e for e in events if e["type"] == "message_update"]
    assert len(updates) > 0


@pytest.mark.slow
@pytest.mark.parametrize("model", ALL_MODELS)
async def test_tool_execution(model):
    """Tool calling — echo tool. Verify tool events and multi-turn."""
    context = AgentContext(
        system_prompt="Use the echo tool to echo the user's message back. After getting the tool result, say 'Done.'",
        messages=[],
        tools=[ECHO_TOOL],
    )
    config = AgentConfig(model=model, convert_to_llm=make_convert_to_llm(model))

    user_msg = {"role": "user", "content": "Echo this: hello world", "timestamp": 0}
    stream = agent_loop([user_msg], context, config)
    events = await collect_events(stream)
    types = event_types(events)

    # Tool execution events present
    assert "tool_execution_start" in types
    assert "tool_execution_end" in types

    # Tool was called correctly
    tool_starts = [e for e in events if e["type"] == "tool_execution_start"]
    assert tool_starts[0]["tool_name"] == "echo"

    tool_ends = [e for e in events if e["type"] == "tool_execution_end"]
    assert not tool_ends[0]["is_error"]
    assert any(
        "hello world" in b.get("text", "") for b in tool_ends[0]["result"]["content"]
    )

    # Final assistant response after tool
    assistant_ends = [
        e
        for e in events
        if e["type"] == "message_end" and e["message"].get("role") == "assistant"
    ]
    # At least 2 assistant messages: one with tool_calls, one with final text
    assert len(assistant_ends) >= 2

    # Last assistant message should have stop_reason="stop" (no more tool calls)
    final = assistant_ends[-1]["message"]
    assert final["stop_reason"] == "stop"
    assert final["content"]  # has text response


@pytest.mark.slow
@pytest.mark.parametrize("model", ALL_MODELS)
async def test_multimodal_spike_detection(model):
    """
    Multimodal: generate bar chart with obvious spike → ask LLM which month.
    Proves: tool calling, image flows through conversation, LLM sees image.
    Chart is dead simple: 6 bars labeled Jan-Jun, one is 10x the others (red, with value labels).
    """
    context = AgentContext(
        system_prompt="You are a data analyst. Use tools when asked. Be concise.",
        messages=[],
        tools=[CHART_TOOL],
    )
    config = AgentConfig(model=model, convert_to_llm=make_convert_to_llm(model))

    # Turn 1: generate the chart
    user_msg = {
        "role": "user",
        "content": "Generate a chart of monthly server errors.",
        "timestamp": 0,
    }
    stream = agent_loop([user_msg], context, config)

    spike_info = {}
    events_t1 = []
    async for event in stream:
        events_t1.append(event)
        if (
            event["type"] == "tool_execution_end"
            and event["tool_name"] == "generate_chart"
        ):
            spike_info = event["result"]["details"]

    assert spike_info.get("spike_month"), (
        f"No spike info captured. Events: {event_types(events_t1)}"
    )

    # Check turn 1 didn't error
    agent_end = next(e for e in events_t1 if e["type"] == "agent_end")
    last_assistant = None
    for msg in reversed(agent_end["messages"]):
        if msg.get("role") == "assistant":
            last_assistant = msg
            break
    assert last_assistant and last_assistant["stop_reason"] not in (
        "error",
        "aborted",
    ), f"Turn 1 failed: {last_assistant}"

    # Carry turn 1 messages into context (agent_loop doesn't mutate caller's context)
    context.messages.extend(agent_end["messages"])

    # Turn 2: ask which month has the spike (LLM must look at image)
    user_msg2 = {
        "role": "user",
        "content": "Which month has the highest error count? Reply with just the month name.",
        "timestamp": 1,
    }
    stream2 = agent_loop([user_msg2], context, config)

    answer_text = ""
    async for event in stream2:
        if (
            event["type"] == "message_update"
            and event.get("delta_type") == "text_delta"
        ):
            delta = event["delta"]
            if delta.get("content"):
                answer_text += delta["content"]

    # Check LLM identified the spike month (Jan/Feb/Mar/Apr/May/Jun)
    spike_month = spike_info["spike_month"].lower()  # e.g. "Mar"
    answer_lower = answer_text.lower()

    # Accept full or abbreviated month name
    month_full = {
        "jan": "january",
        "feb": "february",
        "mar": "march",
        "apr": "april",
        "may": "may",
        "jun": "june",
    }[spike_month.lower()]

    assert spike_month.lower() in answer_lower or month_full in answer_lower, (
        f"Model {model}: expected '{spike_month}' in answer, got: {answer_text[:200]}"
    )


# ── New slow integration tests ────────────────────────────────────────────


async def _always_fail_exec(tool_call_id, params, signal=None, on_update=None):
    raise RuntimeError(params.get("message", "This tool always fails"))


ALWAYS_FAIL_TOOL = Tool(
    name="always_fail",
    description="Always raises an error. Use when the user asks you to test error handling.",
    parameters={
        "type": "object",
        "properties": {
            "message": {"type": "string", "description": "Error message"},
        },
    },
    execute=_always_fail_exec,
)


@pytest.mark.slow
@pytest.mark.parametrize("model", ALL_MODELS)
async def test_tool_error_recovery(model):
    """LLM calls always_fail, sees error, adapts (stops calling or reports failure)."""
    context = AgentContext(
        system_prompt=(
            "You have an always_fail tool. If you call it and it errors, "
            "do NOT retry — just report the failure to the user."
        ),
        messages=[],
        tools=[ALWAYS_FAIL_TOOL],
    )
    config = AgentConfig(model=model, convert_to_llm=make_convert_to_llm(model))

    user_msg = {
        "role": "user",
        "content": "Call always_fail with message 'test error'",
        "timestamp": 0,
    }
    stream = agent_loop([user_msg], context, config)
    events = await collect_events(stream)
    types = event_types(events)

    # Tool was called and errored
    tool_ends = [e for e in events if e["type"] == "tool_execution_end"]
    assert len(tool_ends) >= 1
    assert tool_ends[0]["is_error"] is True

    # Agent finished (didn't crash or infinite loop)
    assert "agent_end" in types
    agent_end = next(e for e in events if e["type"] == "agent_end")
    last_assistant = None
    for msg in reversed(agent_end["messages"]):
        if msg.get("role") == "assistant":
            last_assistant = msg
            break
    assert last_assistant and last_assistant["stop_reason"] == "stop"


THINKING_MODELS = [
    "anthropic/claude-sonnet-4-6",
    "gemini/gemini-3-flash-preview",
]


@pytest.mark.slow
@pytest.mark.parametrize("model", THINKING_MODELS)
async def test_thinking_traces(model):
    """Models with reasoning_effort produce thinking_delta events."""
    context = AgentContext(
        system_prompt="Think step by step, then answer.",
        messages=[],
    )
    config = AgentConfig(
        model=model,
        convert_to_llm=make_convert_to_llm(model),
        reasoning_effort="low",
    )

    user_msg = {"role": "user", "content": "What is 17 * 23?", "timestamp": 0}
    stream = agent_loop([user_msg], context, config)
    events = await collect_events(stream)

    thinking_events = [
        e
        for e in events
        if e["type"] == "message_update" and e.get("delta_type") == "thinking_delta"
    ]
    assert len(thinking_events) > 0, f"No thinking_delta events for {model}"


@pytest.mark.slow
@pytest.mark.parametrize("model", ALL_MODELS)
async def test_usage_present(model):
    """Token usage is non-zero on assistant messages."""
    context = AgentContext(
        system_prompt="Reply briefly.",
        messages=[],
    )
    config = AgentConfig(model=model, convert_to_llm=make_convert_to_llm(model))

    user_msg = {"role": "user", "content": "Say hello.", "timestamp": 0}
    stream = agent_loop([user_msg], context, config)
    events = await collect_events(stream)

    assistant_ends = [
        e
        for e in events
        if e["type"] == "message_end" and e["message"].get("role") == "assistant"
    ]
    assert len(assistant_ends) >= 1
    usage = assistant_ends[0]["message"]["usage"]
    assert usage["prompt_tokens"] > 0
    assert usage["completion_tokens"] > 0
    assert usage["total_tokens"] > 0


@pytest.mark.slow
@pytest.mark.parametrize("model", ALL_MODELS)
async def test_stop_reason_with_tools(model):
    """First assistant message has stop_reason related to tool calling."""
    context = AgentContext(
        system_prompt="Always use the echo tool, then say Done.",
        messages=[],
        tools=[ECHO_TOOL],
    )
    config = AgentConfig(model=model, convert_to_llm=make_convert_to_llm(model))

    user_msg = {"role": "user", "content": "Echo 'test'", "timestamp": 0}
    stream = agent_loop([user_msg], context, config)
    events = await collect_events(stream)

    assistant_ends = [
        e
        for e in events
        if e["type"] == "message_end" and e["message"].get("role") == "assistant"
    ]
    # First assistant message should have tool_calls and appropriate stop_reason
    first = assistant_ends[0]["message"]
    assert first.get("tool_calls") is not None
    assert first["stop_reason"] in ("tool_calls", "stop")


@pytest.mark.slow
@pytest.mark.parametrize("model", ALL_MODELS)
async def test_multi_tool_sequential_live(model):
    """Multiple tool calls execute in order with real LLM."""
    order = []

    async def tracking_echo(tool_call_id, params, signal=None, on_update=None):
        order.append(params.get("message", ""))
        return ToolResult(content=[{"type": "text", "text": params["message"]}])

    tracking_tool = Tool(
        name="echo",
        description="Echo a message back. Call this for each item separately.",
        parameters={
            "type": "object",
            "properties": {
                "message": {"type": "string", "description": "Message to echo"},
            },
            "required": ["message"],
        },
        execute=tracking_echo,
    )
    context = AgentContext(
        system_prompt="Echo each word separately using the echo tool (one call per word). Then say Done.",
        messages=[],
        tools=[tracking_tool],
    )
    config = AgentConfig(model=model, convert_to_llm=make_convert_to_llm(model))

    user_msg = {
        "role": "user",
        "content": "Echo these words: apple banana",
        "timestamp": 0,
    }
    stream = agent_loop([user_msg], context, config)
    events = await collect_events(stream)

    tool_starts = [e for e in events if e["type"] == "tool_execution_start"]
    assert len(tool_starts) >= 2, f"Expected 2+ tool calls, got {len(tool_starts)}"
    assert len(order) >= 2


@pytest.mark.slow
async def test_abort_mid_stream_live():
    """Abort during real streaming → stop_reason='aborted'."""
    signal = asyncio.Event()

    context = AgentContext(
        system_prompt="Write a very long essay about the history of mathematics. Be extremely detailed.",
        messages=[],
    )
    config = AgentConfig(
        model="gemini/gemini-3-flash-preview",
        convert_to_llm=make_convert_to_llm("gemini/gemini-3-flash-preview"),
    )

    user_msg = {"role": "user", "content": "Write the essay now.", "timestamp": 0}
    stream = agent_loop([user_msg], context, config, signal)

    chunk_count = 0
    async for event in stream:
        if event["type"] == "message_update":
            chunk_count += 1
            if chunk_count >= 3:
                signal.set()
                break

    # Stream should have been interrupted
    assert signal.is_set()
    assert chunk_count >= 3


# ── Group 8: Bug fix regression tests ──────────────────────────────────────


async def test_agent_loop_does_not_mutate_context_object(mock_llm):
    """agent_loop should not reassign ctx.messages — use local copy instead."""
    mock_llm([make_chunk(make_delta(content="hi"))], make_final(content="hi"))
    original_messages = [{"role": "system", "content": "old"}]
    ctx = make_context(messages=original_messages)
    user_msg = {"role": "user", "content": "hello", "timestamp": 0}
    stream = agent_loop([user_msg], ctx, make_config())
    await collect_events(stream)

    assert ctx.messages is original_messages


async def test_skip_tool_call_args_are_parsed_dict():
    """_skip_tool_call should parse JSON string args to dict."""
    stream = EventStream()
    tc = {"id": "call_0", "function": {"name": "bash", "arguments": '{"cmd": "ls"}'}}
    _skip_tool_call(tc, stream)
    stream.end()
    events = await collect_events(stream)

    assert isinstance(events[0]["args"], dict)
    assert events[0]["args"] == {"cmd": "ls"}


async def test_safety_net_error_message_has_all_keys(monkeypatch):
    """Safety net error_msg must include tool_calls, thinking_blocks, reasoning_content."""

    async def explode(*args, **kwargs):
        raise TypeError("internal bug")

    monkeypatch.setattr("py_pi_agent.loop.run_loop", explode)
    ctx = make_context()
    user_msg = {"role": "user", "content": "hi", "timestamp": 0}
    stream = agent_loop([user_msg], ctx, make_config())
    events = await collect_events(stream)

    agent_end = next(e for e in events if e["type"] == "agent_end")
    error_msg = agent_end["messages"][0]
    assert "tool_calls" in error_msg and error_msg["tool_calls"] is None
    assert "thinking_blocks" in error_msg and error_msg["thinking_blocks"] is None
    assert "reasoning_content" in error_msg and error_msg["reasoning_content"] is None


async def test_cancelled_error_ends_stream(monkeypatch):
    """CancelledError must not prevent stream.end() — result must resolve."""

    async def raise_cancelled(*args, **kwargs):
        raise asyncio.CancelledError()

    monkeypatch.setattr("py_pi_agent.loop.run_loop", raise_cancelled)
    ctx = make_context()
    user_msg = {"role": "user", "content": "hi", "timestamp": 0}
    stream = agent_loop([user_msg], ctx, make_config())

    try:
        await asyncio.wait_for(stream.result(), timeout=2.0)
    except asyncio.TimeoutError:
        pytest.fail(
            "stream.result() hung after CancelledError — stream.end() was never called"
        )


async def test_events_are_json_serializable(mock_llm_seq):
    """All events emitted by the loop must be JSON-serializable with raw json.dumps."""
    tc_raw = [
        {"id": "call_0", "function": {"name": "echo", "arguments": '{"message":"hi"}'}}
    ]
    mock_llm_seq(
        [
            (
                [
                    make_chunk(
                        make_delta(
                            tool_calls=[make_tc_delta(0, id="call_0", name="echo")]
                        )
                    ),
                    make_chunk(
                        make_delta(
                            tool_calls=[make_tc_delta(0, arguments='{"message":"hi"}')]
                        )
                    ),
                ],
                make_final(tool_calls_raw=tc_raw, finish_reason="tool_calls"),
            ),
            (
                [make_chunk(make_delta(content="Done"))],
                make_final(content="Done"),
            ),
        ]
    )
    ctx = make_context(
        messages=[{"role": "user", "content": "echo hi"}],
        tools=[_simple_tool("echo", echo_exec)],
    )
    stream = EventStream()
    await run_loop(ctx, [], make_config(), None, stream)
    events = await collect_events(stream)

    for event in events:
        try:
            json.dumps(event)
        except (TypeError, ValueError) as e:
            pytest.fail(f"Event type '{event['type']}' not JSON-serializable: {e}")
