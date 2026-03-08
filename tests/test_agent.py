"""
Tests for the Agent class (agent.py).

Fast tests use mocked litellm. Slow tests use real API calls.
"""

import base64
import io

import numpy as np
import pytest

from liteagent.agent import Agent
from liteagent.types import Tool, ToolResult

from tests.mock_helpers import (
    _Obj,
    make_chunk,
    make_delta,
    make_final,
    simple_tool as _simple_tool,
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


@pytest.mark.asyncio
async def test_set_model_refreshes_converter(mock_llm):
    """set_model() rebuilds the default converter so provider behavior changes."""
    # Tool message with an image block — OpenAI needs hoisting, Anthropic doesn't
    tool_msg = {
        "role": "tool",
        "tool_call_id": "tc1",
        "content": [
            {"type": "text", "text": "result"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
        ],
    }

    # Start with Anthropic — no image hoisting
    agent = Agent(model="anthropic/claude-sonnet-4-6")
    converted = agent._convert_to_llm([tool_msg])
    assert len(converted) == 1  # tool message passes through as-is
    assert isinstance(converted[0]["content"], list)  # image blocks preserved inline

    # Switch to OpenAI — should now hoist images
    agent.set_model("openai/gpt-4o")
    converted = agent._convert_to_llm([tool_msg])
    assert len(converted) == 2  # tool message + synthetic user message with image
    assert converted[1]["role"] == "user"

    # Switch back to Anthropic — hoisting should stop
    agent.set_model("anthropic/claude-sonnet-4-6")
    converted = agent._convert_to_llm([tool_msg])
    assert len(converted) == 1


@pytest.mark.asyncio
async def test_set_model_preserves_custom_converter():
    """set_model() does NOT rebuild when a custom converter was provided."""
    calls = []

    def custom_convert(messages):
        calls.append(messages)
        return messages

    agent = Agent(model="anthropic/claude-sonnet-4-6", convert_to_llm=custom_convert)
    agent.set_model("openai/gpt-4o")

    tool_msg = {
        "role": "tool",
        "tool_call_id": "tc1",
        "content": [
            {"type": "text", "text": "result"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
        ],
    }
    converted = agent._convert_to_llm([tool_msg])
    assert len(calls) == 1  # custom converter was called
    assert len(converted) == 1  # no hoisting — custom converter passed through


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


@pytest.mark.asyncio
async def test_unsubscribe_during_emit_does_not_skip_subscribers():
    """Unsubscribing mid-emission must not skip the next subscriber.

    Real-world case: a one-shot listener unsubscribes on agent_end,
    causing the next subscriber (e.g. a logger) to miss that same event.
    """
    from liteagent.agent import Agent

    agent = Agent(model="test-model")
    b_saw_it = []

    unsub_a = None

    def subscriber_a(e):
        unsub_a()  # one-shot: unsubscribe after first event

    def subscriber_b(e):
        b_saw_it.append(e)

    unsub_a = agent.subscribe(subscriber_a)
    agent.subscribe(subscriber_b)

    agent._emit({"type": "agent_end"})

    assert len(b_saw_it) == 1, (
        f"Subscriber B saw {len(b_saw_it)} events, expected 1. "
        "List mutation during _emit skipped subscriber B."
    )


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

    monkeypatch.setattr("liteagent.loop.litellm.acompletion", exploding_acompletion)

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
    "gemini/gemini-3-pro-preview",
    "gemini/gemini-3-flash-preview",
    # "gemini/gemini-3.1-pro-preview",  # unreliable — garbage responses on tool result images
    "gpt-5.2",
    "gpt-5.3-codex",
    "gpt-5.4",
]


@pytest.mark.slow
@pytest.mark.parametrize("model", ALL_MODELS)
async def test_agent_simple_prompt(model):
    """Basic prompt through Agent API with default converter."""
    agent = Agent(model=model, system_prompt="Be concise. One sentence max.")
    await agent.prompt("What is 2 + 2?")

    assistants = [m for m in agent.messages if m.get("role") == "assistant"]
    assert len(assistants) == 1
    assert assistants[0].get("content")
    assert "4" in assistants[0]["content"]


@pytest.mark.slow
@pytest.mark.parametrize("model", ALL_MODELS)
async def test_agent_tool_round_trip(model):
    """Tool execution through Agent API with default converter."""

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
    )
    await agent.prompt("Echo 'hello world'")

    roles = [m.get("role") for m in agent.messages]
    assert "tool" in roles
    assistants = [m for m in agent.messages if m.get("role") == "assistant"]
    assert any(a.get("content") for a in assistants)


@pytest.mark.slow
async def test_agent_multi_turn():
    """Two prompts — second references first turn's context."""
    agent = Agent(
        model="anthropic/claude-sonnet-4-6",
        system_prompt="Be concise. Remember what the user says.",
    )

    await agent.prompt("My favorite color is blue.")
    await agent.prompt("What is my favorite color?")

    last = [m for m in agent.messages if m.get("role") == "assistant"][-1]
    assert "blue" in last["content"].lower()


@pytest.mark.slow
async def test_agent_abort_live():
    """Abort mid-stream — partial message with real content gets preserved."""
    agent = Agent(
        model="anthropic/claude-sonnet-4-6",
        system_prompt="Write a very long essay about the history of computing.",
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

    from liteagent.stream import EventStream

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
        "liteagent.agent.agent_loop",
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
        "liteagent.agent.agent_loop",
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
        "liteagent.agent.agent_loop",
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
        "liteagent.agent.agent_loop",
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

    monkeypatch.setattr("liteagent.loop.litellm.acompletion", exploding_acompletion)

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


# ── Fast tests: continue_run streaming guard ──────────────────────────────


@pytest.mark.asyncio
async def test_continue_run_while_streaming_throws():
    """continue_run() raises if agent is already streaming."""
    agent = Agent(model="test-model")
    agent._state.messages = [{"role": "user", "content": "Hi"}]
    agent._state.is_streaming = True

    with pytest.raises(RuntimeError, match="already processing"):
        await agent.continue_run()

    agent._state.is_streaming = False


# ── Fast tests: follow-up one-at-a-time mode ──────────────────────────────


@pytest.mark.asyncio
async def test_follow_up_one_at_a_time_mode(mock_llm_seq):
    """Default one-at-a-time follow-up mode delivers one message per poll."""
    mock_llm_seq(
        [
            # First: answer the prompt
            (
                [make_chunk(delta=make_delta(content="first"))],
                make_final(content="first"),
            ),
            # Second: answer follow-up 1
            (
                [make_chunk(delta=make_delta(content="second"))],
                make_final(content="second"),
            ),
            # Third: answer follow-up 2
            (
                [make_chunk(delta=make_delta(content="third"))],
                make_final(content="third"),
            ),
        ]
    )

    agent = Agent(model="test-model")  # default follow_up_mode="one-at-a-time"
    agent.follow_up("follow 1")
    agent.follow_up("follow 2")
    await agent.prompt("Start")

    # Both follow-ups should have been delivered (one per outer loop iteration)
    user_msgs = [m for m in agent.messages if m.get("role") == "user"]
    contents = [m["content"] for m in user_msgs]
    assert "follow 1" in contents
    assert "follow 2" in contents
    assert not agent._follow_up_queue

    # Should have 3 assistant responses (prompt + 2 follow-ups)
    assistants = [m for m in agent.messages if m.get("role") == "assistant"]
    assert len(assistants) == 3


# ── Fast tests: clear queue methods ───────────────────────────────────────


@pytest.mark.asyncio
async def test_clear_steering_queue():
    """clear_steering_queue clears only steering, preserves follow-up."""
    agent = Agent(model="test-model")
    agent.steer("s1")
    agent.steer("s2")
    agent.follow_up("f1")

    agent.clear_steering_queue()

    assert not agent._steering_queue
    assert len(agent._follow_up_queue) == 1


@pytest.mark.asyncio
async def test_clear_follow_up_queue():
    """clear_follow_up_queue clears only follow-up, preserves steering."""
    agent = Agent(model="test-model")
    agent.steer("s1")
    agent.follow_up("f1")
    agent.follow_up("f2")

    agent.clear_follow_up_queue()

    assert len(agent._steering_queue) == 1
    assert not agent._follow_up_queue


@pytest.mark.asyncio
async def test_clear_all_queues():
    """clear_all_queues clears both."""
    agent = Agent(model="test-model")
    agent.steer("s1")
    agent.follow_up("f1")

    agent.clear_all_queues()

    assert not agent._steering_queue
    assert not agent._follow_up_queue
    assert not agent.has_queued_messages()


# ── Fast tests: thinking_level reaches litellm ────────────────────────────


@pytest.mark.asyncio
async def test_thinking_level_off_sends_no_reasoning(mock_llm):
    """thinking_level='off' → reasoning_effort not in litellm kwargs."""
    chunks = [make_chunk(delta=make_delta(content="ok"))]
    final = make_final(content="ok")
    captured = mock_llm(chunks, final)

    agent = Agent(model="test-model")
    assert agent.state.thinking_level == "off"
    await agent.prompt("Hi")

    assert "reasoning_effort" not in captured or captured["reasoning_effort"] is None


@pytest.mark.asyncio
async def test_thinking_level_sends_reasoning_effort(mock_llm):
    """thinking_level='high' → reasoning_effort='high' in litellm kwargs."""
    chunks = [make_chunk(delta=make_delta(content="ok"))]
    final = make_final(content="ok")
    captured = mock_llm(chunks, final)

    agent = Agent(model="test-model")
    agent.set_thinking_level("high")
    await agent.prompt("Hi")

    assert captured.get("reasoning_effort") == "high"


# ── Fast tests: transform_context through Agent ──────────────────────────


@pytest.mark.asyncio
async def test_transform_context_called(mock_llm):
    """transform_context hook is called and modifies what the LLM sees."""
    chunks = [make_chunk(delta=make_delta(content="ok"))]
    final = make_final(content="ok")
    captured = mock_llm(chunks, final)

    transform_calls = []

    def my_transform(messages, signal=None):
        transform_calls.append(len(messages))
        # Inject a system-like user message
        return [{"role": "user", "content": "INJECTED"}] + messages

    agent = Agent(
        model="test-model",
        transform_context=my_transform,
    )
    await agent.prompt("Real message")

    # transform_context was called
    assert len(transform_calls) == 1

    # The LLM should have received the injected message
    # (captured["messages"] has system prompt prepended + convert_to_llm output)
    sent_contents = [m.get("content") for m in captured["messages"]]
    assert "INJECTED" in sent_contents


# ── Fast tests: default converter ─────────────────────────────────────────


def test_default_convert_strips_liteagent_metadata():
    """Default converter strips liteagent metadata, passes everything else.
    Non-OpenAI: tool result images stay in the tool message.
    """
    from liteagent.convert import make_default_convert

    convert = make_default_convert("anthropic/claude-sonnet-4-6")
    messages = [
        {
            "role": "assistant",
            "content": "Hello",
            "thinking_blocks": [{"type": "thinking", "thinking": "Let me think..."}],
            "reasoning_content": "I should say hello",
            "provider_specific_fields": {"thought_signatures": ["sig123"]},
            "tool_calls": None,
            "usage": {"prompt_tokens": 10},
            "stop_reason": "stop",
            "error_message": None,
            "timestamp": 12345,
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What is this?"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,abc"},
                },
            ],
            "timestamp": 12346,
        },
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "name": "analyze",
            "content": [
                {"type": "text", "text": "result"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,xyz"},
                },
            ],
            "details": {"extra": "ui-only"},
            "is_error": False,
            "timestamp": 12347,
        },
    ]
    converted = convert(messages)
    assert len(converted) == 3

    # Assistant: thinking/reasoning/provider_specific_fields preserved, metadata stripped
    asst = converted[0]
    assert asst["thinking_blocks"] == [
        {"type": "thinking", "thinking": "Let me think..."}
    ]
    assert asst["reasoning_content"] == "I should say hello"
    assert asst["provider_specific_fields"] == {"thought_signatures": ["sig123"]}
    assert asst["content"] == "Hello"
    assert "usage" not in asst
    assert "stop_reason" not in asst
    assert "timestamp" not in asst

    # User: multimodal content preserved, metadata stripped
    user = converted[1]
    assert len(user["content"]) == 2
    assert user["content"][1]["type"] == "image_url"
    assert "timestamp" not in user

    # Tool: multimodal content preserved (non-OpenAI), metadata stripped
    tool = converted[2]
    assert isinstance(tool["content"], list)
    assert tool["content"][1]["type"] == "image_url"
    assert tool["tool_call_id"] == "call_1"
    assert "details" not in tool
    assert "is_error" not in tool
    assert "timestamp" not in tool


def test_default_convert_openai_hoists_tool_images():
    """OpenAI converter hoists images from tool results into user messages."""
    from liteagent.convert import make_default_convert

    convert = make_default_convert("gpt-5.2")
    messages = [
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": [
                {"type": "text", "text": "Here is the chart."},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,abc123"},
                },
            ],
        },
    ]
    converted = convert(messages)

    # Should produce 2 messages: text-only tool + user message with image
    assert len(converted) == 2

    # Tool message: text only
    assert converted[0]["role"] == "tool"
    assert converted[0]["content"] == "Here is the chart."
    assert converted[0]["tool_call_id"] == "call_1"

    # User message: hoisted image
    assert converted[1]["role"] == "user"
    assert isinstance(converted[1]["content"], list)
    assert converted[1]["content"][1]["type"] == "image_url"
    assert "abc123" in converted[1]["content"][1]["image_url"]["url"]


def test_default_convert_openai_no_hoist_text_only():
    """OpenAI converter doesn't hoist when tool result has no images."""
    from liteagent.convert import make_default_convert

    convert = make_default_convert("gpt-5.2")
    messages = [
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": [{"type": "text", "text": "Just text."}],
        },
    ]
    converted = convert(messages)
    assert len(converted) == 1
    assert converted[0]["role"] == "tool"


def test_default_convert_filters_non_llm_roles():
    """Default converter filters out non-LLM message roles."""
    from liteagent.convert import make_default_convert

    convert = make_default_convert("anthropic/claude-sonnet-4-6")
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "system", "content": "you are helpful"},
        {"role": "assistant", "content": "hello"},
        {"role": "custom_notification", "content": "ignored"},
    ]
    converted = convert(messages)
    assert len(converted) == 2
    assert converted[0]["role"] == "user"
    assert converted[1]["role"] == "assistant"


# ── Fast tests: mode getters/setters ──────────────────────────────────────


def test_steering_mode_getter_setter():
    """set/get_steering_mode round-trip."""
    agent = Agent(model="test-model")
    assert agent.get_steering_mode() == "one-at-a-time"
    agent.set_steering_mode("all")
    assert agent.get_steering_mode() == "all"


def test_follow_up_mode_getter_setter():
    """set/get_follow_up_mode round-trip."""
    agent = Agent(model="test-model")
    assert agent.get_follow_up_mode() == "one-at-a-time"
    agent.set_follow_up_mode("all")
    assert agent.get_follow_up_mode() == "all"


# ── Slow tests: multimodal + default converter ───────────────────────────
#
# These test the default converter through the Agent class with real API calls.
# No custom convert_to_llm — exercises the denylist converter for all providers.
# Covers: multimodal user messages, multimodal tool results, thinking round-trip.


def _make_bar_chart_b64():
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


def _make_scatter_b64():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    np.random.seed(42)
    x = np.append(np.random.normal(50, 10, 100), [90, 92, 88])
    y = np.append(2.3 * x[:100] + np.random.normal(0, 15, 100), [50, 55, 48])
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(x[:100], y[:100], alpha=0.5, c="#3498db")
    ax.scatter(x[100:], y[100:], c="#e74c3c", s=80)
    ax.set_title("Latency vs Memory")
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=72)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _make_heatmap_b64():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    np.random.seed(99)
    data = np.random.poisson(5, (7, 24))
    data[2, 2:5] = [45, 52, 38]  # Wednesday 2-4am hotspot
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.imshow(data, cmap="YlOrRd", aspect="auto")
    ax.set_yticks(range(7), ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    ax.set_xlabel("Hour of Day")
    ax.set_title("Error Rate Heatmap — May 2026")
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=72)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


@pytest.mark.slow
@pytest.mark.parametrize("model", ALL_MODELS)
async def test_agent_multimodal_user_message(model):
    """User sends text + image — model sees the image and responds about it."""
    chart_b64 = _make_bar_chart_b64()
    agent = Agent(model=model, system_prompt="Be concise.")
    await agent.prompt(
        "Which month has the highest value in this chart? Reply with just the month name.",
        images=[
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{chart_b64}"},
            }
        ],
    )

    assistants = [m for m in agent.messages if m.get("role") == "assistant"]
    assert assistants
    answer = assistants[-1].get("content", "").lower()
    assert "may" in answer, f"Expected 'may' in: {answer[:200]}"


@pytest.mark.slow
@pytest.mark.parametrize("model", ALL_MODELS)
async def test_agent_multimodal_tool_result(model):
    """Tool returns text + image — model must prove it saw the image.

    The bar chart has an obvious spike in May (580 vs ~130 baseline).
    The tool returns the chart image. The model must identify "May" from the image.
    """
    chart_b64 = _make_bar_chart_b64()

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

    agent = Agent(
        model=model,
        tools=[chart_tool],
        system_prompt=(
            "Use get_error_chart when asked. After seeing the chart, "
            "answer the user's question about it. Be concise."
        ),
    )
    await agent.prompt(
        "Get the error chart, then tell me: which month has the highest error count? "
        "Reply with just the month name."
    )

    # Must have called the tool
    tool_msgs = [m for m in agent.messages if m.get("role") == "tool"]
    assert tool_msgs, "Expected tool result message"

    # Tool result should have multimodal content (not flattened to text)
    tool_content = tool_msgs[0].get("content")
    assert isinstance(tool_content, list), (
        f"Expected list content, got {type(tool_content)}"
    )
    assert any(b.get("type") == "image_url" for b in tool_content), (
        "Image block missing from tool result"
    )

    # Model must identify May from the chart image
    assistants = [m for m in agent.messages if m.get("role") == "assistant"]
    last = assistants[-1]
    assert last.get("content"), "Expected text response after tool result"
    answer = last["content"].lower()
    assert "may" in answer, (
        f"Model must see the chart image to identify May. Got: {answer[:300]}"
    )


@pytest.mark.slow
@pytest.mark.parametrize(
    "model",
    [
        "anthropic/claude-sonnet-4-6",
        "gemini/gemini-3-flash-preview",
    ],
)
async def test_agent_thinking_round_trip(model):
    """Multi-turn with thinking — signatures survive round-trip via default converter.

    Turn 1: model reasons + responds
    Turn 2: model responds using prior context (requires valid thinking signatures)
    """
    agent = Agent(model=model, system_prompt="Be concise. Show your reasoning.")
    agent.set_thinking_level("high")

    await agent.prompt(
        "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. "
        "How much does the ball cost? Think carefully."
    )

    # Turn 1: check thinking fields exist
    asst1 = [m for m in agent.messages if m.get("role") == "assistant"][0]
    has_thinking = bool(asst1.get("thinking_blocks")) or bool(
        asst1.get("reasoning_content")
    )
    has_psf = bool(asst1.get("provider_specific_fields"))
    # At least one thinking indicator should be present
    assert has_thinking or has_psf, (
        f"Expected thinking content on {model}. "
        f"thinking_blocks={bool(asst1.get('thinking_blocks'))}, "
        f"reasoning_content={bool(asst1.get('reasoning_content'))}, "
        f"provider_specific_fields={bool(asst1.get('provider_specific_fields'))}"
    )

    # Turn 2: this will fail if signatures were dropped by the converter
    await agent.prompt(
        "Now if the bat and ball cost $2.20 total and the bat costs $2.00 more "
        "than the ball, how much does the ball cost?"
    )

    asst2_list = [m for m in agent.messages if m.get("role") == "assistant"]
    assert len(asst2_list) >= 2
    asst2 = asst2_list[-1]
    assert asst2.get("content"), "Expected response on turn 2"


@pytest.mark.slow
@pytest.mark.parametrize("model", ALL_MODELS)
async def test_agent_multimodal_multi_turn(model):
    """4-turn multimodal flow matching investigate.py scenario.

    Turn 1: user sends bar chart (May=580 spike) → model calls tool
    Turn 2: tool returns scatter plot (outliers at high latency) → model responds
    Turn 3: user sends heatmap (Wednesday 2-4am hotspot) → model must see the pattern
    Turn 4: text-only summary → model must reference findings from all 3 images

    Every turn asserts the model actually understood the image content.
    """
    chart_b64 = _make_bar_chart_b64()
    scatter_b64 = _make_scatter_b64()
    heatmap_b64 = _make_heatmap_b64()

    async def analyze_exec(tool_call_id, params, signal=None, on_update=None):
        return ToolResult(
            content=[
                {
                    "type": "text",
                    "text": "See attached scatter plot.",
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{scatter_b64}"},
                },
            ],
        )

    analyze_tool = Tool(
        name="analyze_metrics",
        description="Analyze metrics. Returns a scatter plot image.",
        parameters={
            "type": "object",
            "properties": {"metric": {"type": "string"}},
            "required": ["metric"],
        },
        execute=analyze_exec,
    )

    agent = Agent(
        model=model,
        tools=[analyze_tool],
        system_prompt="Use analyze_metrics when asked to investigate. Be concise.",
    )

    # Turn 1: user sends bar chart → model should call tool
    await agent.prompt(
        "Here's an error chart. Which month has the spike? "
        "Use analyze_metrics to investigate latency for that month.",
        images=[
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{chart_b64}"},
            }
        ],
    )

    tool_msgs = [m for m in agent.messages if m.get("role") == "tool"]
    assert tool_msgs, "Turn 1: expected tool call"

    # Tool result should have multimodal content (image not flattened)
    tool_content = tool_msgs[0].get("content")
    assert isinstance(tool_content, list), "Turn 1: tool content should be list"
    assert any(b.get("type") == "image_url" for b in tool_content), (
        "Turn 1: image block missing from tool result"
    )

    # Turn 2: model saw bar chart in turn 1 — must mention May
    assistants_t2 = [m for m in agent.messages if m.get("role") == "assistant"]
    assert assistants_t2, "Turn 2: expected assistant response"
    # Check across ALL assistant messages (some models mention May before tool call,
    # some after). At least one must reference the spike month.
    all_assistant_text = " ".join(
        a.get("content", "") or "" for a in assistants_t2
    ).lower()
    assert "may" in all_assistant_text, (
        f"Turn 1-2: model must see bar chart and identify May. Got: {all_assistant_text[:300]}"
    )

    # Turn 3: user sends heatmap → model must see Wednesday pattern
    await agent.prompt(
        "Here's an error rate heatmap by hour and day of week for May. "
        "Which day of the week has the concentrated error spike? "
        "Reply mentioning the day name.",
        images=[
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{heatmap_b64}"},
            }
        ],
    )

    assistants_t3 = [m for m in agent.messages if m.get("role") == "assistant"]
    assert len(assistants_t3) > len(assistants_t2), (
        "Turn 3: expected new assistant response"
    )
    t3_text = assistants_t3[-1].get("content", "").lower()
    assert "wednesday" in t3_text or "wed" in t3_text, (
        f"Turn 3: model must see heatmap and identify Wednesday. Got: {t3_text[:300]}"
    )

    # Turn 4: text-only summary — model must remember all images
    await agent.prompt(
        "Summarize your findings from all three visuals. "
        "Mention the spike month, the day-of-week pattern, and the latency anomaly."
    )

    assistants_t4 = [m for m in agent.messages if m.get("role") == "assistant"]
    assert len(assistants_t4) > len(assistants_t3), (
        "Turn 4: expected new assistant response"
    )
    summary = assistants_t4[-1].get("content", "").lower()
    assert summary, "Turn 4: expected text summary"
    # Must reference findings from all 3 images
    assert "may" in summary, f"Turn 4: summary must mention May. Got: {summary[:300]}"
    assert "wednesday" in summary or "wed" in summary, (
        f"Turn 4: summary must mention Wednesday. Got: {summary[:300]}"
    )
    assert "latency" in summary or "outlier" in summary or "memory" in summary, (
        f"Turn 4: summary must mention latency/outlier/memory. Got: {summary[:300]}"
    )
