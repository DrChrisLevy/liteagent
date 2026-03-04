"""
Tests for the Agent class (agent.py).

Fast tests use mocked litellm. Slow tests use real API calls.
"""

import pytest

from py_pi_agent.agent import Agent
from py_pi_agent.types import Tool, ToolResult

# ── Mock infrastructure ───────────────────────────────────────────────────


class _Obj:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def make_delta(**kw):
    return _Obj(**{k: v for k, v in kw.items() if v is not None})


def make_chunk(delta=None, finish_reason=None):
    return _Obj(
        choices=[_Obj(delta=delta or make_delta(), finish_reason=finish_reason)]
    )


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


@pytest.fixture
def mock_llm(monkeypatch):
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


# ── Helpers ───────────────────────────────────────────────────────────────


def _simple_tool(name="test_tool", execute_fn=None):
    async def _default(tool_call_id, params, signal=None, on_update=None):
        return ToolResult(content=[{"type": "text", "text": "ok"}])

    return Tool(
        name=name,
        description=f"Test tool: {name}",
        parameters={"type": "object", "properties": {}},
        execute=execute_fn or _default,
    )


# ── Fast tests: basic prompt ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_prompt_events_and_messages(mock_llm):
    """prompt() delivers events to subscribers and accumulates messages."""
    chunks = [make_chunk(delta=make_delta(content="hello"))]
    final = make_final(content="hello")
    mock_llm(chunks, final)

    events = []
    agent = Agent(model="test-model")
    agent.subscribe(lambda e: events.append(e))
    await agent.prompt("Hi")

    assert any(e["type"] == "agent_end" for e in events)
    assert len(agent.messages) > 0
    assert any(m.get("role") == "assistant" for m in agent.messages)
    assert not agent.state.is_streaming


@pytest.mark.asyncio
async def test_prompt_while_streaming_throws(mock_llm):
    """prompt() raises if agent is already streaming."""
    # We need prompt to not complete immediately — use a task
    chunks = [make_chunk(delta=make_delta(content="hello"))]
    final = make_final(content="hello")
    mock_llm(chunks, final)

    agent = Agent(model="test-model")
    # Manually set streaming to simulate mid-run
    agent._state.is_streaming = True

    with pytest.raises(RuntimeError, match="already processing"):
        await agent.prompt("Second")

    agent._state.is_streaming = False


# ── Fast tests: steering ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_steer_queues_messages(mock_llm_seq):
    """steer() queues messages that get picked up on the initial steering poll."""
    mock_llm_seq(
        [
            (
                [make_chunk(delta=make_delta(content="redirected"))],
                make_final(content="redirected"),
            ),
        ]
    )

    agent = Agent(model="test-model")
    agent.steer("Do this instead")
    await agent.prompt("Original task")

    assert not agent._steering_queue
    user_msgs = [m for m in agent.messages if m.get("role") == "user"]
    contents = [m["content"] for m in user_msgs]
    assert "Do this instead" in contents


# ── Fast tests: follow-up ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_follow_up_after_idle(mock_llm_seq):
    """follow_up() delivered when agent would stop."""
    mock_llm_seq(
        [
            (
                [make_chunk(delta=make_delta(content="first"))],
                make_final(content="first"),
            ),
            (
                [make_chunk(delta=make_delta(content="second"))],
                make_final(content="second"),
            ),
        ]
    )

    agent = Agent(model="test-model")
    agent.follow_up("Now say second")
    await agent.prompt("Say first")

    assistants = [m for m in agent.messages if m.get("role") == "assistant"]
    assert len(assistants) == 2


# ── Fast tests: abort ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_abort_during_prompt(mock_llm):
    """abort() during prompt sets signal and agent finishes cleanly."""
    chunks = [make_chunk(delta=make_delta(content="hel"))]
    final = make_final(content="hel")
    mock_llm(chunks, final)

    agent = Agent(model="test-model")

    def abort_on_first_event(event):
        if event["type"] == "message_start":
            agent.abort()

    agent.subscribe(abort_on_first_event)
    await agent.prompt("Hi")

    assert not agent.state.is_streaming
    assert agent._signal is None  # cleaned up


# ── Fast tests: setters (no streaming guard — matches pi) ────────────────


@pytest.mark.asyncio
async def test_setters_no_streaming_guard():
    """Setters work without streaming guard — pi allows mid-run changes."""
    agent = Agent(model="test-model")
    # These should NOT raise even if we pretend to be streaming
    agent._state.is_streaming = True
    agent.set_model("other-model")
    agent.set_tools([])
    agent.set_system_prompt("new")
    agent.set_thinking_level("high")
    agent._state.is_streaming = False

    assert agent.state.model == "other-model"
    assert agent.state.thinking_level == "high"


@pytest.mark.asyncio
async def test_setters_take_effect_next_run(mock_llm):
    """set_model() changes the model for the next prompt."""
    chunks = [make_chunk(delta=make_delta(content="ok"))]
    final = make_final(content="ok")
    captured = mock_llm(chunks, final)

    agent = Agent(model="model-a")
    await agent.prompt("Hi")
    assert captured["model"] == "model-a"

    agent.set_model("model-b")
    mock_llm(chunks, final)
    await agent.prompt("Hi again")
    assert captured["model"] == "model-b"


# ── Fast tests: reset ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_reset_clears_state(mock_llm):
    """reset() clears messages/queues/error, keeps model/tools."""
    chunks = [make_chunk(delta=make_delta(content="hi"))]
    final = make_final(content="hi")
    mock_llm(chunks, final)

    agent = Agent(model="test-model", tools=[_simple_tool()])
    await agent.prompt("Hi")
    assert len(agent.messages) > 0

    agent.steer("queued")
    agent.follow_up("queued")
    agent.reset()

    assert agent.messages == []
    assert agent.state.error is None
    assert not agent.has_queued_messages()
    assert agent.state.model == "test-model"
    assert len(agent.state.tools) == 1


# ── Fast tests: subscribe ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_subscribe_and_unsubscribe(mock_llm):
    """Subscriber gets events, unsubscribe stops them."""
    chunks = [make_chunk(delta=make_delta(content="hi"))]
    final = make_final(content="hi")
    mock_llm(chunks, final)

    events_received = []
    agent = Agent(model="test-model")
    unsub = agent.subscribe(lambda e: events_received.append(e))

    await agent.prompt("Hi")
    assert len(events_received) > 0

    count_before = len(events_received)
    unsub()

    mock_llm(
        [make_chunk(delta=make_delta(content="bye"))],
        make_final(content="bye"),
    )
    await agent.prompt("Bye")
    assert len(events_received) == count_before


# ── Fast tests: continue_run ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_continue_run_from_tool_result(mock_llm):
    """continue_run() from a tool result message."""
    chunks = [make_chunk(delta=make_delta(content="summary"))]
    final = make_final(content="summary")
    mock_llm(chunks, final)

    agent = Agent(model="test-model")
    agent._state.messages = [
        {"role": "user", "content": "What's the weather?"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "c0", "function": {"name": "weather", "arguments": "{}"}}
            ],
            "stop_reason": "tool_calls",
        },
        {
            "role": "tool",
            "tool_call_id": "c0",
            "content": [{"type": "text", "text": "72°F sunny"}],
            "is_error": False,
        },
    ]

    events = []
    agent.subscribe(lambda e: events.append(e))
    await agent.continue_run()

    assert any(e["type"] == "agent_end" for e in events)


@pytest.mark.asyncio
async def test_continue_run_empty_throws():
    """No messages → ValueError."""
    agent = Agent(model="test-model")
    with pytest.raises(ValueError, match="no messages"):
        await agent.continue_run()


@pytest.mark.asyncio
async def test_continue_run_assistant_tail_steering(mock_llm):
    """Assistant last + steering queue → uses steering."""
    chunks = [make_chunk(delta=make_delta(content="ok"))]
    final = make_final(content="ok")
    mock_llm(chunks, final)

    agent = Agent(model="test-model")
    agent._state.messages = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!", "stop_reason": "stop"},
    ]
    agent.steer("Do something else")

    await agent.continue_run()

    assert not agent._steering_queue
    user_msgs = [m for m in agent.messages if m.get("role") == "user"]
    contents = [m["content"] for m in user_msgs]
    assert "Do something else" in contents


@pytest.mark.asyncio
async def test_continue_run_assistant_tail_follow_up(mock_llm):
    """Assistant last + follow-up queue → uses follow-up messages."""
    chunks = [make_chunk(delta=make_delta(content="ok"))]
    final = make_final(content="ok")
    mock_llm(chunks, final)

    agent = Agent(model="test-model")
    agent._state.messages = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!", "stop_reason": "stop"},
    ]
    agent.follow_up("Also do this")

    await agent.continue_run()

    assert not agent._follow_up_queue


@pytest.mark.asyncio
async def test_continue_run_assistant_tail_empty_throws():
    """Assistant last + empty queues → ValueError."""
    agent = Agent(model="test-model")
    agent._state.messages = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!", "stop_reason": "stop"},
    ]
    with pytest.raises(ValueError, match="queued messages"):
        await agent.continue_run()


# ── Fast tests: queue modes ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_queue_modes_one_at_a_time():
    """one-at-a-time dequeue returns one message."""
    agent = Agent(model="test-model")
    agent.steer("msg1")
    agent.steer("msg2")
    agent.steer("msg3")

    result = agent._dequeue_steering()
    assert len(result) == 1
    assert result[0]["content"] == "msg1"
    assert len(agent._steering_queue) == 2


@pytest.mark.asyncio
async def test_queue_modes_all():
    """'all' mode dequeue returns all messages."""
    agent = Agent(model="test-model", steering_mode="all")
    agent.steer("msg1")
    agent.steer("msg2")
    agent.steer("msg3")

    result = agent._dequeue_steering()
    assert len(result) == 3
    assert len(agent._steering_queue) == 0


# ── Fast tests: state updates from events ────────────────────────────────


@pytest.mark.asyncio
async def test_messages_appended_on_message_end(mock_llm):
    """Messages appended on each message_end, not batched (like pi)."""
    chunks = [make_chunk(delta=make_delta(content="hi"))]
    final = make_final(content="hi")
    mock_llm(chunks, final)

    messages_during_run = []
    agent = Agent(model="test-model")

    def track(event):
        if event["type"] == "message_end":
            messages_during_run.append(len(agent.messages))

    agent.subscribe(track)
    await agent.prompt("Hi")

    # Messages should have grown incrementally
    assert len(messages_during_run) >= 2  # user + assistant
    for i in range(1, len(messages_during_run)):
        assert messages_during_run[i] > messages_during_run[i - 1]


@pytest.mark.asyncio
async def test_state_updates_from_events(mock_llm_seq):
    """stream_message and pending_tool_calls tracked correctly."""
    tool = _simple_tool("echo")
    tc = [{"id": "c0", "function": {"name": "echo", "arguments": "{}"}}]

    mock_llm_seq(
        [
            ([], make_final(tool_calls_raw=tc, finish_reason="tool_calls")),
            (
                [make_chunk(delta=make_delta(content="done"))],
                make_final(content="done"),
            ),
        ]
    )

    agent = Agent(model="test-model", tools=[tool])
    await agent.prompt("Call echo")

    assert not agent.state.is_streaming
    assert agent.state.stream_message is None
    assert len(agent.state.pending_tool_calls) == 0


@pytest.mark.asyncio
async def test_finally_cleanup(monkeypatch):
    """is_streaming/stream_message/pending_tool_calls reset even on error."""

    async def exploding_acompletion(**kwargs):
        raise Exception("LLM exploded")

    monkeypatch.setattr("py_pi_agent.loop.litellm.acompletion", exploding_acompletion)

    agent = Agent(model="test-model")
    await agent.prompt("Hi")

    assert not agent.state.is_streaming
    assert agent.state.stream_message is None
    assert len(agent.state.pending_tool_calls) == 0
    assert agent.state.error is not None


# ── Fast tests: multi-turn ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_multi_turn_messages_accumulate(mock_llm):
    """Two prompts → messages from both turns accumulate."""
    chunks = [make_chunk(delta=make_delta(content="reply"))]
    final = make_final(content="reply")
    mock_llm(chunks, final)

    agent = Agent(model="test-model")
    await agent.prompt("Turn 1")
    count_after_t1 = len(agent.messages)

    mock_llm(chunks, final)
    await agent.prompt("Turn 2")
    count_after_t2 = len(agent.messages)

    assert count_after_t2 > count_after_t1


# ── Fast tests: prompt overloads ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_prompt_with_string(mock_llm):
    """prompt(str) converts to user message."""
    mock_llm(
        [make_chunk(delta=make_delta(content="ok"))],
        make_final(content="ok"),
    )
    agent = Agent(model="test-model")
    await agent.prompt("hello")

    user_msgs = [m for m in agent.messages if m.get("role") == "user"]
    assert user_msgs[0]["content"] == "hello"


@pytest.mark.asyncio
async def test_prompt_with_dict(mock_llm):
    """prompt(dict) uses dict as-is."""
    mock_llm(
        [make_chunk(delta=make_delta(content="ok"))],
        make_final(content="ok"),
    )
    agent = Agent(model="test-model")
    await agent.prompt({"role": "user", "content": "from dict"})

    user_msgs = [m for m in agent.messages if m.get("role") == "user"]
    assert user_msgs[0]["content"] == "from dict"


@pytest.mark.asyncio
async def test_prompt_with_list(mock_llm):
    """prompt(list) injects multiple messages."""
    mock_llm(
        [make_chunk(delta=make_delta(content="ok"))],
        make_final(content="ok"),
    )
    agent = Agent(model="test-model")
    await agent.prompt(
        [
            {"role": "user", "content": "first"},
            {"role": "user", "content": "second"},
        ]
    )

    user_msgs = [m for m in agent.messages if m.get("role") == "user"]
    assert len(user_msgs) == 2
    assert user_msgs[0]["content"] == "first"
    assert user_msgs[1]["content"] == "second"


@pytest.mark.asyncio
async def test_prompt_with_images(mock_llm):
    """prompt(str, images=[...]) builds multimodal user message."""
    mock_llm(
        [make_chunk(delta=make_delta(content="ok"))],
        make_final(content="ok"),
    )
    agent = Agent(model="test-model")
    images = [{"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}}]
    await agent.prompt("describe this", images=images)

    user_msgs = [m for m in agent.messages if m.get("role") == "user"]
    content = user_msgs[0]["content"]
    assert isinstance(content, list)
    assert content[0] == {"type": "text", "text": "describe this"}
    assert content[1]["type"] == "image_url"


# ── Slow tests: real API ─────────────────────────────────────────────────

ALL_MODELS = [
    "anthropic/claude-sonnet-4-6",
    "anthropic/claude-opus-4-6",
    "gemini/gemini-3-flash-preview",
    # "gemini/gemini-3.1-pro-preview",  # consistently timing out
    "gpt-5.2",
]


def _make_convert(model):
    def convert(messages):
        result = []
        for m in messages:
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
                    content = "\n".join(text_parts)
                result.append(
                    {
                        "role": "tool",
                        "tool_call_id": m["tool_call_id"],
                        "content": content,
                    }
                )
        return result

    return convert


@pytest.mark.slow
@pytest.mark.parametrize("model", ALL_MODELS)
async def test_agent_simple_prompt(model):
    """Basic prompt through Agent API."""
    agent = Agent(
        model=model,
        system_prompt="Be concise. One sentence max.",
        convert_to_llm=_make_convert(model),
    )
    await agent.prompt("What is 2 + 2?")

    assistants = [m for m in agent.messages if m.get("role") == "assistant"]
    assert len(assistants) == 1
    assert assistants[0].get("content")
    assert "4" in assistants[0]["content"]


@pytest.mark.slow
@pytest.mark.parametrize("model", ALL_MODELS)
async def test_agent_tool_round_trip(model):
    """Tool execution through Agent API."""

    async def echo_exec(tool_call_id, params, signal=None, on_update=None):
        return ToolResult(content=[{"type": "text", "text": params["message"]}])

    echo_tool = Tool(
        name="echo",
        description="Echo back a message",
        parameters={
            "type": "object",
            "properties": {"message": {"type": "string"}},
            "required": ["message"],
        },
        execute=echo_exec,
    )

    agent = Agent(
        model=model,
        tools=[echo_tool],
        system_prompt="Use the echo tool when asked. Be concise.",
        convert_to_llm=_make_convert(model),
    )
    await agent.prompt("Echo 'hello world'")

    roles = [m.get("role") for m in agent.messages]
    assert "tool" in roles
    assistants = [m for m in agent.messages if m.get("role") == "assistant"]
    assert any(a.get("content") for a in assistants)


@pytest.mark.slow
async def test_agent_multi_turn():
    """Two prompts — second references first turn's context."""
    model = "anthropic/claude-sonnet-4-6"
    agent = Agent(
        model=model,
        system_prompt="Be concise. Remember what the user says.",
        convert_to_llm=_make_convert(model),
    )

    await agent.prompt("My favorite color is blue.")
    await agent.prompt("What is my favorite color?")

    last = [m for m in agent.messages if m.get("role") == "assistant"][-1]
    assert "blue" in last["content"].lower()


@pytest.mark.slow
async def test_agent_abort_live():
    """Abort mid-stream — partial message with real content gets preserved."""

    model = "anthropic/claude-sonnet-4-6"
    agent = Agent(
        model=model,
        system_prompt="Write a very long essay about the history of computing.",
        convert_to_llm=_make_convert(model),
    )

    chunk_count = 0

    def on_event(event):
        nonlocal chunk_count
        if (
            event["type"] == "message_update"
            and event.get("delta_type") == "text_delta"
        ):
            chunk_count += 1
            if chunk_count >= 5:
                agent.abort()

    agent.subscribe(on_event)
    await agent.prompt("Go ahead, write the essay.")

    assert not agent.state.is_streaming
    # After abort, the last assistant message should exist (partial content preserved)
    # or agent.state.error should be set (empty scaffolding → "Request was aborted")
    assistants = [m for m in agent.messages if m.get("role") == "assistant"]
    if assistants:
        last = assistants[-1]
        # Either has real content, or stop_reason reflects the abort/error
        assert last.get("content") or last.get("stop_reason") in ("aborted", "error")
    else:
        # No assistant message means the abort raised and error was captured
        assert agent.state.error is not None


# ── Fast tests: partial preservation on abort ────────────────────────────


def _make_partial_stream(partial_msg):
    """Build an EventStream that emits message_start without message_end.

    This simulates an interrupted stream where the loop crashes between
    message_start and message_end — the edge case the partial preservation
    code in _run_loop guards against (same as pi agent.ts:504-518).
    """
    import asyncio

    from py_pi_agent.stream import EventStream

    stream = EventStream()

    async def _run():
        stream.push({"type": "agent_start"})
        stream.push({"type": "turn_start"})
        stream.push({"type": "message_start", "message": partial_msg})
        # No message_end — simulates interrupted stream
        stream.push({"type": "agent_end", "messages": []})
        stream.end([])

    asyncio.get_running_loop().create_task(_run())
    return stream


@pytest.mark.asyncio
async def test_partial_with_named_tool_call_preserved(monkeypatch):
    """Interrupted stream: partial with a named tool call is preserved."""
    partial_msg = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {"id": "c0", "function": {"name": "read_file", "arguments": ""}}
        ],
    }

    monkeypatch.setattr(
        "py_pi_agent.agent.agent_loop",
        lambda *a, **kw: _make_partial_stream(partial_msg),
    )

    agent = Agent(model="test-model")
    await agent.prompt("Hi")

    assistants = [m for m in agent.messages if m.get("role") == "assistant"]
    assert any(m.get("tool_calls") for m in assistants)


@pytest.mark.asyncio
async def test_partial_with_empty_scaffold_discarded(monkeypatch):
    """Interrupted stream: partial with empty tool-call scaffold is discarded."""
    partial_msg = {
        "role": "assistant",
        "content": None,
        "tool_calls": [{"id": "", "function": {"name": "", "arguments": ""}}],
    }

    monkeypatch.setattr(
        "py_pi_agent.agent.agent_loop",
        lambda *a, **kw: _make_partial_stream(partial_msg),
    )

    agent = Agent(model="test-model")
    agent._signal = None  # ensure abort path triggers
    await agent.prompt("Hi")

    # Empty scaffold should NOT have been preserved
    assistants = [m for m in agent.messages if m.get("role") == "assistant"]
    for m in assistants:
        if m.get("tool_calls"):
            for tc in m["tool_calls"]:
                assert (tc.get("function", {}).get("name") or "").strip()


@pytest.mark.asyncio
async def test_partial_with_reasoning_preserved(monkeypatch):
    """Interrupted stream: partial with reasoning_content is preserved."""
    partial_msg = {
        "role": "assistant",
        "content": None,
        "reasoning_content": "Let me think about this...",
    }

    monkeypatch.setattr(
        "py_pi_agent.agent.agent_loop",
        lambda *a, **kw: _make_partial_stream(partial_msg),
    )

    agent = Agent(model="test-model")
    await agent.prompt("Hi")

    assistants = [m for m in agent.messages if m.get("role") == "assistant"]
    assert any(m.get("reasoning_content") for m in assistants)


@pytest.mark.asyncio
async def test_partial_whitespace_reasoning_discarded(monkeypatch):
    """Interrupted stream: partial with whitespace-only reasoning is discarded."""
    partial_msg = {
        "role": "assistant",
        "content": None,
        "reasoning_content": "   \n  ",
    }

    monkeypatch.setattr(
        "py_pi_agent.agent.agent_loop",
        lambda *a, **kw: _make_partial_stream(partial_msg),
    )

    agent = Agent(model="test-model")
    await agent.prompt("Hi")

    # Whitespace-only reasoning should NOT be preserved
    assistants = [m for m in agent.messages if m.get("role") == "assistant"]
    for m in assistants:
        rc = m.get("reasoning_content")
        if rc:
            assert rc.strip(), "Whitespace-only reasoning should not be preserved"


# ── Fast tests: steering during tool execution ───────────────────────────


@pytest.mark.asyncio
async def test_steering_skips_remaining_tools(mock_llm_seq):
    """Steering mid-tool-execution skips remaining tools."""
    call_log = []
    agent = None  # forward reference

    async def tool_a_exec(tool_call_id, params, signal=None, on_update=None):
        call_log.append("a")
        # Inject steering DURING tool_a execution — should skip tool_b
        agent.steer("Stop and do this instead")
        return ToolResult(content=[{"type": "text", "text": "a done"}])

    async def tool_b_exec(tool_call_id, params, signal=None, on_update=None):
        call_log.append("b")
        return ToolResult(content=[{"type": "text", "text": "b done"}])

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

    tc_raw = [
        {"id": "c0", "function": {"name": "tool_a", "arguments": "{}"}},
        {"id": "c1", "function": {"name": "tool_b", "arguments": "{}"}},
    ]

    # Need real tool call chunks so the loop sees tool_calls on the assistant message
    tc_chunk = make_chunk(
        delta=_Obj(
            content=None,
            reasoning_content=None,
            thinking_blocks=None,
            tool_calls=[
                _Obj(index=0, id="c0", function=_Obj(name="tool_a", arguments="{}")),
                _Obj(index=1, id="c1", function=_Obj(name="tool_b", arguments="{}")),
            ],
        ),
        finish_reason="tool_calls",
    )

    mock_llm_seq(
        [
            # First turn: assistant calls both tools
            ([tc_chunk], make_final(tool_calls_raw=tc_raw, finish_reason="tool_calls")),
            # Second turn: after steering, assistant responds
            (
                [make_chunk(delta=make_delta(content="redirected"))],
                make_final(content="redirected"),
            ),
        ]
    )

    agent = Agent(model="test-model", tools=[tool_a, tool_b])
    await agent.prompt("Call both tools")

    # tool_a ran, tool_b should have been skipped
    assert "a" in call_log
    assert "b" not in call_log

    # Steering message should appear in history
    user_msgs = [m for m in agent.messages if m.get("role") == "user"]
    contents = [m["content"] for m in user_msgs]
    assert "Stop and do this instead" in contents


# ── Fast tests: follow-up "all" mode ─────────────────────────────────────


@pytest.mark.asyncio
async def test_follow_up_all_mode(mock_llm_seq):
    """'all' follow-up mode delivers all queued messages at once."""
    mock_llm_seq(
        [
            (
                [make_chunk(delta=make_delta(content="first"))],
                make_final(content="first"),
            ),
            (
                [make_chunk(delta=make_delta(content="got them"))],
                make_final(content="got them"),
            ),
        ]
    )

    agent = Agent(model="test-model", follow_up_mode="all")
    agent.follow_up("follow 1")
    agent.follow_up("follow 2")
    await agent.prompt("Start")

    # Both follow-ups should have been delivered
    user_msgs = [m for m in agent.messages if m.get("role") == "user"]
    contents = [m["content"] for m in user_msgs]
    assert "follow 1" in contents
    assert "follow 2" in contents
    assert not agent._follow_up_queue


# ── Fast tests: error message structure ──────────────────────────────────


@pytest.mark.asyncio
async def test_error_message_structure(monkeypatch):
    """Error message has correct fields: stop_reason, error_message, usage, timestamp, model."""

    async def exploding_acompletion(**kwargs):
        raise Exception("boom")

    monkeypatch.setattr("py_pi_agent.loop.litellm.acompletion", exploding_acompletion)

    agent = Agent(model="test-model")
    await agent.prompt("Hi")

    error_msgs = [
        m
        for m in agent.messages
        if m.get("role") == "assistant" and m.get("error_message")
    ]
    assert len(error_msgs) >= 1
    err = error_msgs[-1]
    assert err["stop_reason"] == "error"
    assert "boom" in err["error_message"]
    assert err["model"] == "test-model"
    assert "prompt_tokens" in err["usage"]
    assert isinstance(err["timestamp"], int)


# ── Fast tests: wait_for_idle ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_wait_for_idle_when_idle():
    """wait_for_idle returns immediately when not running."""
    agent = Agent(model="test-model")
    await agent.wait_for_idle()  # should not hang


@pytest.mark.asyncio
async def test_wait_for_idle_during_run(mock_llm):
    """wait_for_idle resolves after prompt completes."""
    chunks = [make_chunk(delta=make_delta(content="hi"))]
    final = make_final(content="hi")
    mock_llm(chunks, final)

    agent = Agent(model="test-model")
    await agent.prompt("Hi")
    # After prompt returns, wait_for_idle should resolve immediately
    await agent.wait_for_idle()
    assert not agent.state.is_streaming


# ── Fast tests: replace_messages ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_replace_messages():
    """replace_messages replaces full history."""
    agent = Agent(model="test-model")
    agent.append_message({"role": "user", "content": "old"})
    assert len(agent.messages) == 1

    new_msgs = [
        {"role": "user", "content": "new1"},
        {"role": "assistant", "content": "new2"},
    ]
    agent.replace_messages(new_msgs)
    assert len(agent.messages) == 2
    assert agent.messages[0]["content"] == "new1"

    # Should be a copy, not a reference
    new_msgs.append({"role": "user", "content": "extra"})
    assert len(agent.messages) == 2


# ── Fast tests: clear_messages vs reset ───────────────────────────────────


@pytest.mark.asyncio
async def test_clear_messages_preserves_queues_and_error():
    """clear_messages only clears messages, not queues or error."""
    agent = Agent(model="test-model")
    agent.append_message({"role": "user", "content": "hi"})
    agent.steer("queued steering")
    agent.follow_up("queued follow-up")
    agent._state.error = "some error"

    agent.clear_messages()

    assert agent.messages == []
    assert agent.has_queued_messages()  # queues preserved
    assert agent.state.error == "some error"  # error preserved


@pytest.mark.asyncio
async def test_reset_clears_everything_except_config():
    """reset clears messages, queues, and error. Keeps model/tools/system_prompt."""
    agent = Agent(model="test-model", system_prompt="sys", tools=[_simple_tool()])
    agent.append_message({"role": "user", "content": "hi"})
    agent.steer("queued")
    agent.follow_up("queued")
    agent._state.error = "some error"

    agent.reset()

    assert agent.messages == []
    assert not agent.has_queued_messages()
    assert agent.state.error is None
    # Config preserved
    assert agent.state.model == "test-model"
    assert agent.state.system_prompt == "sys"
    assert len(agent.state.tools) == 1
