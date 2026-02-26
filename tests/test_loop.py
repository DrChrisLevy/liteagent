"""
Tests for the core loop (loop.py).

Unit tests run by default. Slow tests (real API calls) need: ./dev test -m slow
Multimodal spike detection runs against all 5 target models.
"""

import asyncio
import base64
import io

import numpy as np
import pytest

from py_pi_agent.loop import agent_loop, agent_loop_continue
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
    """Generate a time series chart with a spike. Returns text + image."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from datetime import datetime, timedelta

    days = int(params.get("days", 90))
    title = params.get("title", "Server Response Time (ms)")

    # Generate data with a random spike
    base = 120 + np.cumsum(np.random.randn(days) * 2)
    base = np.clip(base, 50, 250)
    spike_idx = np.random.randint(days // 4, 3 * days // 4)
    spike_magnitude = np.random.uniform(150, 400)
    base[spike_idx] += spike_magnitude
    if spike_idx > 0:
        base[spike_idx - 1] += spike_magnitude * 0.3
    if spike_idx < days - 1:
        base[spike_idx + 1] += spike_magnitude * 0.2

    dates = [datetime(2025, 1, 1) + timedelta(days=i) for i in range(days)]
    spike_date = dates[spike_idx].strftime("%B %d")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(dates, base, linewidth=1.5)
    ax.set_title(title)
    ax.set_ylabel("ms")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    plt.xticks(rotation=45)
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    img_b64 = base64.b64encode(buf.getvalue()).decode()

    return ToolResult(
        content=[
            {"type": "text", "text": f"Generated chart: {title} ({days} days)"},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{img_b64}"},
            },
        ],
        details={"spike_date": spike_date, "spike_idx": spike_idx},
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
        "hello world" in b.get("text", "") for b in tool_ends[0]["result"].content
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
    Multimodal test: generate chart with spike → ask LLM to identify spike date.
    Proves: tool calling, image content flows through conversation, LLM sees images.
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
        "content": "Plot server response times for the last 90 days.",
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
            spike_info = event["result"].details

    # Chart was generated
    assert spike_info.get("spike_date"), (
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

    # Turn 2: ask about the spike (LLM must look at image in history)
    user_msg2 = {
        "role": "user",
        "content": "I see there's a spike in the chart. What date does it occur on? Reply with just the date.",
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
            if delta.content:
                answer_text += delta.content

    # Verify LLM identified the spike date (lenient — chart reading is approximate)
    spike_date = spike_info["spike_date"]  # e.g. "March 04"
    parts = spike_date.split()
    month_full = parts[0].lower()  # "march"
    month_abbr = month_full[:3]  # "mar"
    day = int(parts[1]) if len(parts) > 1 else 0

    answer_lower = answer_text.lower()

    # Accept: full month name, abbreviated, or numeric month in ISO date
    month_num = {
        "january": "01",
        "february": "02",
        "march": "03",
        "april": "04",
        "may": "05",
        "june": "06",
        "july": "07",
        "august": "08",
        "september": "09",
        "october": "10",
        "november": "11",
        "december": "12",
    }.get(month_full, "")
    month_found = (
        month_full in answer_lower
        or month_abbr in answer_lower
        or f"-{month_num}-" in answer_lower
    )
    assert month_found, (
        f"Model {model}: expected month '{month_full}' in answer, got: {answer_text[:200]}"
    )

    # Accept off-by-one day (chart reading is approximate, +-2 days)
    nearby_days = {str(day + offset) for offset in range(-2, 3) if day + offset > 0}
    day_found = any(d in answer_lower for d in nearby_days)
    assert day_found, (
        f"Model {model}: expected day ~{day} in answer, got: {answer_text[:200]}"
    )
