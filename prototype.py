"""
py-pi-agent in one file. Minimal Python port of pi-mono's agent loop.
Uses litellm for LLM calls, Pydantic for tool validation, asyncio for concurrency.
"""

import asyncio
import time
import json
from dataclasses import dataclass, field
from typing import Callable

import litellm
from dotenv import load_dotenv

load_dotenv()
litellm.modify_params = True  # auto-fix role alternation, orphaned tool calls, etc.


# ── EventStream ─────────────────────────────────────────────────────────────
# Async producer-consumer queue. The loop pushes events, consumers iterate.
# Faithful port of pi's EventStream<T, R> from event-stream.ts.


class EventStream:
    def __init__(self, on_push: Callable = None):
        self._queue: asyncio.Queue[dict | None] = asyncio.Queue()
        self._result: list | None = None
        self._done = asyncio.Event()
        self._on_push = on_push

    def push(self, event: dict):
        if self._on_push:
            self._on_push(event)
        self._queue.put_nowait(event)

    def end(self, result: list):
        self._result = result
        self._done.set()
        self._queue.put_nowait(None)  # sentinel

    async def __aiter__(self):
        while True:
            event = await self._queue.get()
            if event is None:
                return
            yield event

    async def result(self) -> list:
        await self._done.wait()
        return self._result


# ── Types ───────────────────────────────────────────────────────────────────


@dataclass
class ToolResult:
    content: list[dict]  # [{"type": "text", "text": ...}, {"type": "image_url", ...}]
    details: dict = field(default_factory=dict)


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict  # JSON Schema
    execute: (
        Callable  # async def(tool_call_id, params, signal, on_update) -> ToolResult
    )
    label: str = ""
    params_model: type = None  # Optional Pydantic BaseModel for validation


@dataclass
class AgentConfig:
    model: str
    convert_to_llm: Callable  # AgentMessage[] -> LLM messages[]
    transform_context: Callable = None
    get_steering_messages: Callable = None
    get_follow_up_messages: Callable = None
    reasoning_effort: str = None  # "low", "medium", "high" — None = no thinking
    max_tokens: int = None
    temperature: float = None
    max_retry_delay_ms: int = 60000


@dataclass
class AgentState:
    system_prompt: str = ""
    model: str = ""
    thinking_level: str = "off"
    tools: list = field(default_factory=list)
    messages: list = field(default_factory=list)
    is_streaming: bool = False
    stream_message: dict | None = None
    pending_tool_calls: set = field(default_factory=set)
    error: str | None = None


# ── Tool Validation ─────────────────────────────────────────────────────────
# Pydantic validates + coerces (LLM sends "42" → 42). Equivalent of pi's AJV.


def validate_tool_args(tool: Tool, raw_args: dict) -> dict:
    if tool.params_model is None:
        return raw_args  # no model = no validation, pass through
    validated = tool.params_model(**raw_args)
    return validated.model_dump()


# ── Loop ────────────────────────────────────────────────────────────────────
# The dual while-loop. Faithful port of agent-loop.ts's runLoop().


async def run_loop(
    context: dict,
    new_messages: list,
    config: AgentConfig,
    signal: asyncio.Event,
    stream: EventStream,
):
    first_turn = True
    pending = (
        (await config.get_steering_messages()) if config.get_steering_messages else []
    )

    # Outer loop: follow-ups after agent would stop
    while True:
        has_tool_calls = True
        steering_after_tools = None

        # Inner loop: LLM calls + tool execution + steering
        while has_tool_calls or pending:
            if not first_turn:
                stream.push({"type": "turn_start"})
            else:
                first_turn = False

            # Inject pending messages
            if pending:
                for msg in pending:
                    stream.push({"type": "message_start", "message": msg})
                    stream.push({"type": "message_end", "message": msg})
                    context["messages"].append(msg)
                    new_messages.append(msg)
                pending = []

            # Stream LLM response
            assistant_msg = await stream_llm_response(context, config, signal, stream)
            new_messages.append(assistant_msg)

            if assistant_msg.get("stop_reason") in ("error", "aborted"):
                stream.push(
                    {"type": "turn_end", "message": assistant_msg, "tool_results": []}
                )
                return new_messages

            # Check for tool calls
            tool_calls = assistant_msg.get("tool_calls") or []
            has_tool_calls = len(tool_calls) > 0

            tool_results = []
            if has_tool_calls:
                exec_result = await execute_tool_calls(
                    context.get("tools", []),
                    assistant_msg,
                    signal,
                    stream,
                    config.get_steering_messages,
                )
                tool_results = exec_result["tool_results"]
                steering_after_tools = exec_result.get("steering_messages")

                for tr in tool_results:
                    context["messages"].append(tr)
                    new_messages.append(tr)

            stream.push(
                {
                    "type": "turn_end",
                    "message": assistant_msg,
                    "tool_results": tool_results,
                }
            )

            # Check steering
            if steering_after_tools:
                pending = steering_after_tools
                steering_after_tools = None
            elif config.get_steering_messages:
                pending = await config.get_steering_messages()
            else:
                pending = []

        # Agent would stop — check follow-ups
        follow_ups = (
            (await config.get_follow_up_messages())
            if config.get_follow_up_messages
            else []
        )
        if follow_ups:
            pending = follow_ups
            continue
        break

    return new_messages


async def stream_llm_response(
    context: dict, config: AgentConfig, signal: asyncio.Event, stream: EventStream
) -> dict:
    messages = context["messages"]
    if config.transform_context:
        messages = await config.transform_context(messages)

    llm_messages = config.convert_to_llm(messages)

    # Build litellm kwargs
    kwargs = dict(
        model=config.model,
        messages=llm_messages,
        stream=True,
        stream_options={"include_usage": True},
    )
    if context.get("tools"):
        kwargs["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            }
            for t in context["tools"]
        ]
    if config.reasoning_effort:
        kwargs["reasoning_effort"] = config.reasoning_effort
    if config.max_tokens:
        kwargs["max_tokens"] = config.max_tokens
    if config.temperature is not None:
        kwargs["temperature"] = config.temperature

    # System prompt
    if context.get("system_prompt"):
        kwargs["messages"] = [
            {"role": "system", "content": context["system_prompt"]}
        ] + kwargs["messages"]

    # Stream the response
    text_content = ""
    reasoning_content = ""
    thinking_blocks = []
    tool_calls_map = {}  # index -> {id, name, arguments_str}
    usage = {}
    finish_reason = None

    partial_msg = {"role": "assistant", "content": None, "tool_calls": None}
    stream.push({"type": "message_start", "message": partial_msg})

    try:
        response = await litellm.acompletion(**kwargs)

        async for chunk in response:
            if signal.is_set():
                break

            choices = getattr(chunk, "choices", None) or []
            choice = choices[0] if choices else None

            if choice:
                delta = choice.delta
                if choice.finish_reason:
                    finish_reason = choice.finish_reason

                # Text delta
                if delta.content:
                    text_content += delta.content
                    stream.push(
                        {
                            "type": "message_update",
                            "message": partial_msg,
                            "delta": delta,
                            "delta_type": "text_delta",
                        }
                    )

                # Reasoning delta
                if getattr(delta, "reasoning_content", None):
                    reasoning_content += delta.reasoning_content
                    stream.push(
                        {
                            "type": "message_update",
                            "message": partial_msg,
                            "delta": delta,
                            "delta_type": "thinking_delta",
                        }
                    )

                # Thinking blocks (Anthropic only — includes cryptographic signatures)
                delta_blocks = getattr(delta, "thinking_blocks", None)
                if delta_blocks:
                    thinking_blocks.extend(delta_blocks)

                # Tool call delta
                if delta.tool_calls:
                    for tc_chunk in delta.tool_calls:
                        idx = tc_chunk.index
                        if idx not in tool_calls_map:
                            tool_calls_map[idx] = {
                                "id": "",
                                "name": "",
                                "arguments": "",
                            }
                        if tc_chunk.id:
                            tool_calls_map[idx]["id"] = tc_chunk.id
                        if tc_chunk.function:
                            if tc_chunk.function.name:
                                tool_calls_map[idx]["name"] = tc_chunk.function.name
                            if tc_chunk.function.arguments:
                                tool_calls_map[idx]["arguments"] += (
                                    tc_chunk.function.arguments
                                )
                    stream.push(
                        {
                            "type": "message_update",
                            "message": partial_msg,
                            "delta": delta,
                            "delta_type": "tool_call_delta",
                        }
                    )

            # Usage on final chunk
            chunk_usage = getattr(chunk, "usage", None)
            if chunk_usage:
                details = getattr(chunk_usage, "prompt_tokens_details", None)
                usage = {
                    "prompt_tokens": getattr(chunk_usage, "prompt_tokens", 0) or 0,
                    "completion_tokens": getattr(chunk_usage, "completion_tokens", 0)
                    or 0,
                    "total_tokens": getattr(chunk_usage, "total_tokens", 0) or 0,
                    "cache_read_tokens": getattr(details, "cached_tokens", 0) or 0
                    if details
                    else 0,
                    "cache_creation_tokens": getattr(
                        details, "cache_creation_tokens", 0
                    )
                    or 0
                    if details
                    else 0,
                }

    except Exception as e:
        assistant_msg = {
            "role": "assistant",
            "content": str(e),
            "tool_calls": None,
            "stop_reason": "error",
            "usage": {},
            "timestamp": int(time.time() * 1000),
        }
        stream.push({"type": "message_end", "message": assistant_msg})
        context["messages"].append(assistant_msg)
        return assistant_msg

    # Build finalized assistant message
    tool_calls_list = None
    if tool_calls_map:
        tool_calls_list = [
            {
                "id": tc["id"],
                "type": "function",
                "function": {"name": tc["name"], "arguments": tc["arguments"]},
            }
            for tc in sorted(tool_calls_map.values(), key=lambda x: x["id"])
        ]

    stop_reason = "aborted" if signal.is_set() else (finish_reason or "stop")

    assistant_msg = {
        "role": "assistant",
        "content": text_content or None,
        "tool_calls": tool_calls_list,
        "reasoning_content": reasoning_content or None,
        "thinking_blocks": thinking_blocks or None,
        "usage": usage,
        "stop_reason": stop_reason,
        "timestamp": int(time.time() * 1000),
    }

    stream.push({"type": "message_end", "message": assistant_msg})
    context["messages"].append(assistant_msg)
    return assistant_msg


async def execute_tool_calls(
    tools: list[Tool],
    assistant_msg: dict,
    signal: asyncio.Event,
    stream: EventStream,
    get_steering_messages: Callable = None,
) -> dict:
    tool_calls = assistant_msg.get("tool_calls") or []
    results = []
    steering_messages = None
    tools_by_name = {t.name: t for t in tools}

    for i, tc in enumerate(tool_calls):
        func = tc["function"]
        tool_name = func["name"]
        tool_call_id = tc["id"]
        raw_args = (
            json.loads(func["arguments"])
            if isinstance(func["arguments"], str)
            else func["arguments"]
        )

        stream.push(
            {
                "type": "tool_execution_start",
                "tool_call_id": tool_call_id,
                "tool_name": tool_name,
                "args": raw_args,
            }
        )

        is_error = False
        try:
            tool = tools_by_name.get(tool_name)
            if not tool:
                raise Exception(f"Tool '{tool_name}' not found")

            validated_args = validate_tool_args(tool, raw_args)

            def make_on_update(tid, tname):
                def on_update(partial):
                    stream.push(
                        {
                            "type": "tool_execution_update",
                            "tool_call_id": tid,
                            "tool_name": tname,
                            "partial": partial,
                        }
                    )

                return on_update

            result = await tool.execute(
                tool_call_id,
                validated_args,
                signal,
                make_on_update(tool_call_id, tool_name),
            )
        except Exception as e:
            result = ToolResult(content=[{"type": "text", "text": str(e)}])
            is_error = True

        stream.push(
            {
                "type": "tool_execution_end",
                "tool_call_id": tool_call_id,
                "tool_name": tool_name,
                "result": result,
                "is_error": is_error,
            }
        )

        tool_result_msg = {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": tool_name,
            "content": result.content,
            "is_error": is_error,
            "timestamp": int(time.time() * 1000),
        }
        results.append(tool_result_msg)
        stream.push({"type": "message_start", "message": tool_result_msg})
        stream.push({"type": "message_end", "message": tool_result_msg})

        # Check for steering — skip remaining tools if user interrupted
        if get_steering_messages:
            steering = await get_steering_messages()
            if steering:
                steering_messages = steering
                for skipped_tc in tool_calls[i + 1 :]:
                    results.append(skip_tool_call(skipped_tc, stream))
                break

    return {"tool_results": results, "steering_messages": steering_messages}


def skip_tool_call(tc: dict, stream: EventStream) -> dict:
    tool_call_id = tc["id"]
    tool_name = tc["function"]["name"]
    stream.push(
        {
            "type": "tool_execution_start",
            "tool_call_id": tool_call_id,
            "tool_name": tool_name,
            "args": {},
        }
    )
    stream.push(
        {
            "type": "tool_execution_end",
            "tool_call_id": tool_call_id,
            "tool_name": tool_name,
            "result": ToolResult(
                content=[
                    {"type": "text", "text": "Skipped due to queued user message."}
                ]
            ),
            "is_error": True,
        }
    )
    msg = {
        "role": "tool",
        "tool_call_id": tool_call_id,
        "name": tool_name,
        "content": [{"type": "text", "text": "Skipped due to queued user message."}],
        "is_error": True,
        "timestamp": int(time.time() * 1000),
    }
    stream.push({"type": "message_start", "message": msg})
    stream.push({"type": "message_end", "message": msg})
    return msg


# ── Agent (stateful wrapper) ───────────────────────────────────────────────
# Port of agent.ts's Agent class. Wraps the loop with state management.


def default_convert_to_llm(messages: list) -> list:
    """Keep only LLM-compatible messages."""
    result = []
    for m in messages:
        if m["role"] == "assistant":
            msg = {"role": "assistant"}
            if m.get("content"):
                msg["content"] = m["content"]
            if m.get("tool_calls"):
                msg["tool_calls"] = m["tool_calls"]
            result.append(msg)
        elif m["role"] == "user":
            result.append({"role": "user", "content": m["content"]})
        elif m["role"] == "tool":
            # Convert our tool results to litellm format
            # Keep text and image_url blocks (both are LLM-compatible)
            content = m["content"]
            if isinstance(content, list):
                llm_blocks = [
                    b for b in content if b.get("type") in ("text", "image_url")
                ]
                # litellm expects string content for tool results with text only
                if all(b["type"] == "text" for b in llm_blocks):
                    content = "\n".join(b["text"] for b in llm_blocks)
                else:
                    content = llm_blocks  # mixed text+images: pass as content array
            result.append(
                {"role": "tool", "tool_call_id": m["tool_call_id"], "content": content}
            )
    return result


class Agent:
    def __init__(
        self,
        model: str,
        tools: list[Tool] = None,
        system_prompt: str = "",
        convert_to_llm: Callable = None,
        transform_context: Callable = None,
        steering_mode: str = "one-at-a-time",
        follow_up_mode: str = "one-at-a-time",
        thinking_level: str = "off",
    ):
        self._state = AgentState(
            system_prompt=system_prompt,
            model=model,
            thinking_level=thinking_level,
            tools=tools or [],
        )
        self._convert_to_llm = convert_to_llm or default_convert_to_llm
        self._transform_context = transform_context
        self._steering_mode = steering_mode
        self._follow_up_mode = follow_up_mode
        self._steering_queue: list = []
        self._follow_up_queue: list = []
        self._listeners: list[Callable] = []
        self._signal: asyncio.Event | None = None
        self._idle_event: asyncio.Event = asyncio.Event()
        self._idle_event.set()

    @property
    def state(self) -> AgentState:
        return self._state

    @property
    def messages(self) -> list:
        return self._state.messages

    def subscribe(self, fn: Callable) -> Callable:
        """Subscribe to events. Returns unsubscribe function."""
        self._listeners.append(fn)
        return lambda: self._listeners.remove(fn)

    def _handle_event(self, event: dict):
        """Called on every event — updates Agent state, then notifies subscribers."""
        t = event["type"]
        if t == "message_start":
            self._state.stream_message = event["message"]
        elif t == "message_update":
            self._state.stream_message = event["message"]
        elif t == "message_end":
            self._state.stream_message = None
        elif t == "tool_execution_start":
            self._state.pending_tool_calls = self._state.pending_tool_calls | {
                event["tool_call_id"]
            }
        elif t == "tool_execution_end":
            self._state.pending_tool_calls = self._state.pending_tool_calls - {
                event["tool_call_id"]
            }
        elif t == "turn_end":
            msg = event.get("message", {})
            if msg.get("role") == "assistant" and msg.get("error_message"):
                self._state.error = msg["error_message"]

        for fn in self._listeners:
            fn(event)

    # ── State mutators ──

    def set_system_prompt(self, v: str):
        self._state.system_prompt = v

    def set_model(self, m: str):
        self._state.model = m

    def set_tools(self, t: list[Tool]):
        self._state.tools = t

    def set_thinking_level(self, level: str):
        self._state.thinking_level = level

    # ── Queue management ──

    def steer(self, message: str):
        self._steering_queue.append(
            {"role": "user", "content": message, "timestamp": int(time.time() * 1000)}
        )

    def follow_up(self, message: str):
        self._follow_up_queue.append(
            {"role": "user", "content": message, "timestamp": int(time.time() * 1000)}
        )

    def clear_steering_queue(self):
        self._steering_queue.clear()

    def clear_follow_up_queue(self):
        self._follow_up_queue.clear()

    def clear_all_queues(self):
        self._steering_queue.clear()
        self._follow_up_queue.clear()

    def has_queued_messages(self) -> bool:
        return bool(self._steering_queue or self._follow_up_queue)

    def _dequeue_steering(self) -> list:
        if not self._steering_queue:
            return []
        if self._steering_mode == "one-at-a-time":
            return [self._steering_queue.pop(0)]
        msgs = self._steering_queue[:]
        self._steering_queue.clear()
        return msgs

    def _dequeue_follow_ups(self) -> list:
        if not self._follow_up_queue:
            return []
        if self._follow_up_mode == "one-at-a-time":
            return [self._follow_up_queue.pop(0)]
        msgs = self._follow_up_queue[:]
        self._follow_up_queue.clear()
        return msgs

    # ── Actions ──

    def prompt(self, message: str) -> EventStream:
        if self._state.is_streaming:
            raise RuntimeError(
                "Agent is already processing. Use steer() or follow_up()."
            )

        user_msg = {
            "role": "user",
            "content": message,
            "timestamp": int(time.time() * 1000),
        }
        return self._run_loop([user_msg])

    def continue_run(self) -> EventStream:
        if self._state.is_streaming:
            raise RuntimeError("Agent is already processing.")
        if not self._state.messages:
            raise RuntimeError("No messages to continue from.")

        last = self._state.messages[-1]
        if last["role"] == "assistant":
            steering = self._dequeue_steering()
            if steering:
                return self._run_loop(steering, skip_initial_steering=True)
            follow_ups = self._dequeue_follow_ups()
            if follow_ups:
                return self._run_loop(follow_ups)
            raise RuntimeError(
                "Cannot continue from assistant message without queued messages."
            )

        return self._run_loop(None)

    def abort(self):
        if self._signal:
            self._signal.set()

    async def wait_for_idle(self):
        await self._idle_event.wait()

    def reset(self):
        self._state.messages.clear()
        self._state.is_streaming = False
        self._state.stream_message = None
        self._state.pending_tool_calls = set()
        self._state.error = None
        self._steering_queue.clear()
        self._follow_up_queue.clear()

    # ── Internal ──

    def _run_loop(
        self, messages: list | None, skip_initial_steering: bool = False
    ) -> EventStream:
        stream = EventStream(on_push=self._handle_event)
        self._signal = asyncio.Event()
        self._state.is_streaming = True
        self._state.stream_message = None
        self._state.error = None
        self._idle_event.clear()

        reasoning = (
            None if self._state.thinking_level == "off" else self._state.thinking_level
        )
        _skip = skip_initial_steering

        async def get_steering():
            nonlocal _skip
            if _skip:
                _skip = False
                return []
            return self._dequeue_steering()

        async def get_follow_ups():
            return self._dequeue_follow_ups()

        config = AgentConfig(
            model=self._state.model,
            convert_to_llm=self._convert_to_llm,
            transform_context=self._transform_context,
            get_steering_messages=get_steering,
            get_follow_up_messages=get_follow_ups,
            reasoning_effort=reasoning,
        )

        context = {
            "system_prompt": self._state.system_prompt,
            "messages": self._state.messages[:],
            "tools": self._state.tools,
        }

        async def run():
            try:
                if messages is not None:
                    # New prompt — add messages to context
                    new_msgs = list(messages)
                    for m in messages:
                        context["messages"].append(m)

                    stream.push({"type": "agent_start"})
                    stream.push({"type": "turn_start"})
                    for m in messages:
                        stream.push({"type": "message_start", "message": m})
                        stream.push({"type": "message_end", "message": m})

                    new_msgs = await run_loop(
                        context, new_msgs, config, self._signal, stream
                    )
                else:
                    # Continue — context already has messages
                    new_msgs = []
                    stream.push({"type": "agent_start"})
                    stream.push({"type": "turn_start"})
                    new_msgs = await run_loop(
                        context, new_msgs, config, self._signal, stream
                    )

                # Sync messages back BEFORE ending stream (avoids race with consumer)
                self._state.messages = context["messages"]
                stream.push({"type": "agent_end", "messages": new_msgs})
                stream.end(new_msgs)

            except Exception as e:
                error_msg = {
                    "role": "assistant",
                    "content": "",
                    "stop_reason": "aborted" if self._signal.is_set() else "error",
                    "error_message": str(e),
                    "usage": {},
                    "timestamp": int(time.time() * 1000),
                }
                self._state.messages.append(error_msg)
                self._state.error = str(e)
                stream.push({"type": "agent_end", "messages": [error_msg]})
                stream.end([error_msg])

            finally:
                self._state.is_streaming = False
                self._state.stream_message = None
                self._state.pending_tool_calls = set()
                self._signal = None
                self._idle_event.set()

        # Fire and forget — consumer iterates the stream
        asyncio.create_task(run())

        return stream


# ── Demo Tools ──────────────────────────────────────────────────────────────


async def echo_exec(tool_call_id, params, signal=None, on_update=None):
    """Echo back a message."""
    return ToolResult(content=[{"type": "text", "text": params["message"]}])


async def query_sales_exec(tool_call_id, params, signal=None, on_update=None):
    """Return mock sales data for a given quarter."""
    import random

    quarter = params.get("quarter", "Q1")
    regions = ["North", "South", "East", "West"]
    data = {r: random.randint(50_000, 500_000) for r in regions}
    total = sum(data.values())
    lines = [f"Sales for {quarter}:"]
    for region, amount in data.items():
        lines.append(f"  {region}: ${amount:,}")
    lines.append(f"  Total: ${total:,}")
    return ToolResult(content=[{"type": "text", "text": "\n".join(lines)}])


async def plot_timeseries_exec(tool_call_id, params, signal=None, on_update=None):
    """Generate a time series plot with a random spike. Returns image."""
    import numpy as np
    import seaborn as sns
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import io
    import base64
    from datetime import datetime, timedelta

    days = int(params.get("days", 90))
    title = params.get("title", "Server Response Time (ms)")

    # Generate smooth-ish base data
    np.random.seed(None)  # truly random each time
    base = 120 + np.cumsum(np.random.randn(days) * 2)
    base = np.clip(base, 50, 250)

    # Insert a spike at a random position
    spike_idx = np.random.randint(days // 4, 3 * days // 4)
    spike_magnitude = np.random.uniform(150, 400)
    base[spike_idx] += spike_magnitude
    # Small bleed into neighbors
    if spike_idx > 0:
        base[spike_idx - 1] += spike_magnitude * 0.3
    if spike_idx < days - 1:
        base[spike_idx + 1] += spike_magnitude * 0.2

    dates = [datetime(2025, 1, 1) + timedelta(days=i) for i in range(days)]
    spike_date = dates[spike_idx].strftime("%B %d")

    # Plot with seaborn
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.lineplot(x=dates, y=base, ax=ax, linewidth=1.5)
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
            {
                "type": "text",
                "text": f"Generated time series plot: {title} ({days} days)",
            },
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{img_b64}"},
            },
        ],
        details={
            "spike_date": spike_date,
            "spike_idx": spike_idx,
        },  # ground truth (not sent to LLM)
    )


DEMO_TOOLS = [
    Tool(
        name="echo",
        description="Echo back a message",
        parameters={
            "type": "object",
            "properties": {"message": {"type": "string"}},
            "required": ["message"],
        },
        execute=echo_exec,
    ),
    Tool(
        name="query_sales",
        description="Get sales data for a quarter (Q1, Q2, Q3, Q4)",
        parameters={
            "type": "object",
            "properties": {
                "quarter": {"type": "string", "enum": ["Q1", "Q2", "Q3", "Q4"]}
            },
            "required": ["quarter"],
        },
        execute=query_sales_exec,
    ),
    Tool(
        name="plot_timeseries",
        description="Plot a time series chart of server response times. Returns an image. The data has real anomalies you can analyze.",
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
        execute=plot_timeseries_exec,
    ),
]

SYSTEM_PROMPT = """You are a data analyst assistant. You have these tools:
- echo: Echo back a message
- query_sales: Get sales data for a given quarter
- plot_timeseries: Plot server response times (returns an image with real data)

When asked to analyze charts, look carefully at the image and describe what you see.
When asked about sales, use query_sales.
Be concise."""

# Target models:
#   anthropic/claude-opus-4-6, anthropic/claude-sonnet-4-6,
#   gemini/gemini-3-flash-preview, gemini/gemini-3.1-pro-preview, gpt-5.2
DEFAULT_MODEL = "anthropic/claude-opus-4-6"


# ── Test Harness ────────────────────────────────────────────────────────────


async def test_multimodal(model: str) -> dict:
    """Test: plot a chart, then ask about the spike. Returns ground truth + LLM answer."""
    agent = Agent(model=model, tools=DEMO_TOOLS, system_prompt=SYSTEM_PROMPT)

    # Turn 1: generate the plot
    spike_info = {}
    async for event in agent.prompt("plot server response times for the last 90 days"):
        if (
            event["type"] == "tool_execution_end"
            and event["tool_name"] == "plot_timeseries"
        ):
            spike_info = event["result"].details  # ground truth

    # Turn 2: ask about the spike (LLM must look at the image)
    answer = ""
    async for event in agent.prompt(
        "I see there's a spike in the chart. What date does it occur on?"
    ):
        if (
            event["type"] == "message_update"
            and event.get("delta_type") == "text_delta"
        ):
            delta = event["delta"]
            if delta.content:
                answer += delta.content

    return {
        "model": model,
        "spike_date": spike_info.get("spike_date", "?"),
        "answer": answer[:200],
    }


async def test_all_models():
    """Run multimodal test across all target models."""
    models = [
        "anthropic/claude-opus-4-6",
        "anthropic/claude-sonnet-4-6",
        "gemini/gemini-3-flash-preview",
        "gemini/gemini-3.1-pro-preview",
        "gpt-5.2",
    ]
    print("Multimodal spike detection test\n")
    results = await asyncio.gather(*[test_multimodal(m) for m in models])
    for r in results:
        spike = r["spike_date"]
        answer = r["answer"].lower()
        # Check for month + day match (lenient: "March 04" matches "march 4", "march 4th", etc.)
        parts = spike.split()
        month = parts[0].lower() if parts else ""
        day = str(int(parts[1])) if len(parts) > 1 else ""  # "04" -> "4"
        found = month in answer and day in answer
        status = "PASS" if found else "MISS"
        print(f"  [{status}] {r['model']}")
        print(f"         spike={spike}  answer={r['answer'][:120]}...")
        print()


# ── Interactive Mode ────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if "--test" in sys.argv:
        asyncio.run(test_all_models())
    else:

        async def main():
            model = DEFAULT_MODEL
            for i, arg in enumerate(sys.argv[1:]):
                if arg == "--model" and i + 2 < len(sys.argv):
                    model = sys.argv[i + 2]

            agent = Agent(model=model, tools=DEMO_TOOLS, system_prompt=SYSTEM_PROMPT)
            print(f"Model: {model}")
            print(f"Tools: {', '.join(t.name for t in DEMO_TOOLS)}")
            print("Type a message (ctrl+c to quit)\n")

            while True:
                try:
                    user_input = input("You: ")
                except (KeyboardInterrupt, EOFError):
                    print("\nBye!")
                    break

                async for event in agent.prompt(user_input):
                    t = event["type"]
                    if (
                        t == "message_update"
                        and event.get("delta_type") == "text_delta"
                    ):
                        delta = event["delta"]
                        if delta.content:
                            print(delta.content, end="", flush=True)
                    elif t == "tool_execution_start":
                        print(f"\n[tool] {event['tool_name']}({event['args']})")
                    elif t == "tool_execution_end":
                        print(f"[tool] -> {'error' if event['is_error'] else 'ok'}")
                    elif t == "agent_end":
                        print()

        asyncio.run(main())
