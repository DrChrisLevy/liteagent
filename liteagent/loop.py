"""
Dual while loop with streaming + steering.

Two entry points (agent_loop, agent_loop_continue) spawn an async task that
runs the dual loop engine (run_loop). The engine streams LLM responses via
litellm, executes tools sequentially, handles steering/follow-up injection,
and emits events through an EventStream.

Faithful port of pi-mono's agent-loop.ts.
"""

import asyncio
import copy
import inspect
import json

import litellm

from .stream import EventStream
from .types import (
    AgentContext,
    ToolResult,
    _build_assistant_message,
    _build_tool_result_message,
    _extract_usage,
)

# Auto-fix provider message format issues:
# - inserts placeholder user messages for alternating roles (Anthropic, Bedrock)
# - adds dummy tool definition when messages contain tool_call but no tools param
# - drops thinking param when assistant messages lack thinking_blocks
# - sanitizes orphaned tool calls/results
litellm.modify_params = True


# ── Helpers ────────────────────────────────────────────────────────────────


async def _maybe_await(fn, *args):
    result = fn(*args)
    if inspect.isawaitable(result):
        return await result
    return result


def _validate_tool_args(tool, args_str):
    args = json.loads(args_str) if isinstance(args_str, str) else args_str
    if not tool.params_model:
        return args
    validated = tool.params_model(**args)
    return validated.model_dump()


# ── Skip Tool Call ─────────────────────────────────────────────────────────


def _skip_tool_call(tool_call, stream):
    tc_id = tool_call["id"]
    tc_name = tool_call["function"]["name"]
    tc_args = tool_call["function"]["arguments"]
    if isinstance(tc_args, str):
        try:
            tc_args = json.loads(tc_args)
        except (json.JSONDecodeError, TypeError):
            pass
    result = ToolResult(
        content=[{"type": "text", "text": "Skipped due to queued user message."}],
        details={},
    )

    stream.push(
        {
            "type": "tool_execution_start",
            "tool_call_id": tc_id,
            "tool_name": tc_name,
            "args": tc_args,
        }
    )
    stream.push(
        {
            "type": "tool_execution_end",
            "tool_call_id": tc_id,
            "tool_name": tc_name,
            "result": {"content": result.content, "details": result.details},
            "is_error": True,
        }
    )

    msg = _build_tool_result_message(tc_id, tc_name, result, is_error=True)
    stream.push({"type": "message_start", "message": msg})
    stream.push({"type": "message_end", "message": msg})
    return msg


# ── Tool Execution ─────────────────────────────────────────────────────────


async def execute_tool_calls(
    tools, assistant_msg, signal, stream, get_steering_messages
):
    tool_calls = assistant_msg.get("tool_calls") or []
    results = []
    steering_messages = None
    tools_by_name = {t.name: t for t in (tools or [])}

    for i, tc in enumerate(tool_calls):
        func = tc["function"]
        tc_id = tc["id"]
        tc_name = func["name"]

        # Parse args — may be JSON string or already a dict
        try:
            raw_args = (
                json.loads(func["arguments"])
                if isinstance(func["arguments"], str)
                else func["arguments"]
            )
        except (json.JSONDecodeError, TypeError):
            raw_args = func["arguments"]

        stream.push(
            {
                "type": "tool_execution_start",
                "tool_call_id": tc_id,
                "tool_name": tc_name,
                "args": raw_args,
            }
        )

        is_error = False
        try:
            # Re-parse if initial parse failed (raw_args is still a string)
            if isinstance(raw_args, str):
                raise ValueError(f"Invalid JSON in tool arguments: {raw_args[:100]}")

            tool = tools_by_name.get(tc_name)
            if not tool:
                raise ValueError(f"Tool '{tc_name}' not found")

            validated_args = _validate_tool_args(tool, raw_args)

            # Closure factory — captures correct tc_id/tc_name per iteration
            def _make_on_update(call_id, name, args):
                def on_update(partial):
                    stream.push(
                        {
                            "type": "tool_execution_update",
                            "tool_call_id": call_id,
                            "tool_name": name,
                            "args": args,
                            "partial": {
                                "content": partial.content,
                                "details": partial.details,
                            },
                        }
                    )

                return on_update

            result = await tool.execute(
                tc_id, validated_args, signal, _make_on_update(tc_id, tc_name, raw_args)
            )
        except Exception as e:
            result = ToolResult(content=[{"type": "text", "text": str(e)}], details={})
            is_error = True

        stream.push(
            {
                "type": "tool_execution_end",
                "tool_call_id": tc_id,
                "tool_name": tc_name,
                "result": {"content": result.content, "details": result.details},
                "is_error": is_error,
            }
        )

        msg = _build_tool_result_message(tc_id, tc_name, result, is_error)
        results.append(msg)
        stream.push({"type": "message_start", "message": msg})
        stream.push({"type": "message_end", "message": msg})

        # Check steering after each tool — skip remaining if user interrupted
        if get_steering_messages:
            steering = await _maybe_await(get_steering_messages)
            if steering:
                steering_messages = steering
                for skipped in tool_calls[i + 1 :]:
                    results.append(_skip_tool_call(skipped, stream))
                break

    return {"tool_results": results, "steering_messages": steering_messages}


# ── Stream LLM Response ───────────────────────────────────────────────────


async def stream_llm_response(context, config, signal, stream):
    # 1. Transform context (compaction, pruning)
    messages = context.messages
    if config.transform_context:
        messages = await _maybe_await(config.transform_context, messages, signal)

    # 2. Convert to LLM format
    llm_messages = await _maybe_await(config.convert_to_llm, messages)

    # 3. Prepend system prompt
    if context.system_prompt:
        llm_messages = [
            {"role": "system", "content": context.system_prompt}
        ] + llm_messages

    # 4. Build tool schemas
    tool_schemas = None
    if context.tools:
        tool_schemas = [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": copy.deepcopy(t.parameters),
                },
            }
            for t in context.tools
        ]

    # 5. Build litellm kwargs (only non-None params)
    kwargs = {
        "model": config.model,
        "messages": llm_messages,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if tool_schemas:
        kwargs["tools"] = tool_schemas
    if config.reasoning_effort is not None:
        kwargs["reasoning_effort"] = config.reasoning_effort
    if config.max_tokens is not None:
        kwargs["max_tokens"] = config.max_tokens
    if config.temperature is not None:
        kwargs["temperature"] = config.temperature
    if config.num_retries is not None:
        kwargs["num_retries"] = config.num_retries

    # 6. Streaming state
    # Partial accumulators — for message_update events during streaming
    content_parts = []
    reasoning_parts = []
    tool_calls_acc = {}  # index -> {"id", "function": {"name", "arguments"}}
    message_started = False
    partial = {"role": "assistant", "content": None, "tool_calls": None}
    # Chunks collected for litellm.stream_chunk_builder (builds finalized response)
    chunks = []
    # Capture finish_reason from chunks — stream_chunk_builder loses it (litellm bug)
    chunk_finish_reason = None

    try:
        response = await litellm.acompletion(**kwargs)

        async for chunk in response:
            if signal and signal.is_set():
                break

            chunks.append(chunk)

            choices = getattr(chunk, "choices", None)
            choice = choices[0] if choices else None
            if not choice:
                continue

            delta = choice.delta

            # Capture finish_reason from chunks (stream_chunk_builder loses it)
            if choice.finish_reason:
                chunk_finish_reason = choice.finish_reason

            # Extract delta fields (not all exist on every chunk)
            delta_content = getattr(delta, "content", None)
            delta_rc = getattr(delta, "reasoning_content", None)
            delta_tb = getattr(delta, "thinking_blocks", None)
            delta_tc = getattr(delta, "tool_calls", None)

            # Emit message_start on first substantive delta
            if not message_started:
                if delta_content or delta_rc or delta_tb or delta_tc:
                    message_started = True
                    stream.push({"type": "message_start", "message": {**partial}})

            # Text delta
            if delta_content:
                content_parts.append(delta_content)
                partial["content"] = "".join(content_parts)
                stream.push(
                    {
                        "type": "message_update",
                        "message": {**partial},
                        "delta": {"content": delta_content},
                        "delta_type": "text_delta",
                    }
                )

            # Reasoning content (universal, all providers)
            if delta_rc:
                reasoning_parts.append(delta_rc)
                partial["reasoning_content"] = "".join(reasoning_parts)
                stream.push(
                    {
                        "type": "message_update",
                        "message": {**partial},
                        "delta": {"reasoning_content": delta_rc},
                        "delta_type": "thinking_delta",
                    }
                )

            # Thinking blocks (Anthropic — includes cryptographic signatures)
            if delta_tb:
                # Emit thinking_delta (skip if reasoning_content already emitted for this chunk)
                if not delta_rc:
                    stream.push(
                        {
                            "type": "message_update",
                            "message": {**partial},
                            "delta": {"thinking_blocks": delta_tb},
                            "delta_type": "thinking_delta",
                        }
                    )

            # Tool call deltas — merge by index for partial message snapshots
            if delta_tc:
                for tc_delta in delta_tc:
                    idx = tc_delta.index
                    if idx not in tool_calls_acc:
                        tool_calls_acc[idx] = {
                            "id": "",
                            "type": "function",
                            "function": {"name": "", "arguments": ""},
                        }
                    acc = tool_calls_acc[idx]
                    if tc_delta.id:
                        acc["id"] = tc_delta.id
                    if tc_delta.function:
                        if tc_delta.function.name:
                            acc["function"]["name"] += tc_delta.function.name
                        if tc_delta.function.arguments:
                            acc["function"]["arguments"] += tc_delta.function.arguments

                # Snapshot tool calls — deep copy so later delta mutations
                # don't leak into already-queued events
                partial["tool_calls"] = [
                    {
                        "id": tool_calls_acc[k]["id"],
                        "type": "function",
                        "function": {
                            "name": tool_calls_acc[k]["function"]["name"],
                            "arguments": tool_calls_acc[k]["function"]["arguments"],
                        },
                    }
                    for k in sorted(tool_calls_acc)
                ]
                # Convert tool call deltas to plain dicts
                tc_delta_dicts = []
                for tc_d in delta_tc:
                    d = {"index": tc_d.index}
                    if tc_d.id:
                        d["id"] = tc_d.id
                    if tc_d.function:
                        func = {}
                        if tc_d.function.name:
                            func["name"] = tc_d.function.name
                        if tc_d.function.arguments:
                            func["arguments"] = tc_d.function.arguments
                        if func:
                            d["function"] = func
                    tc_delta_dicts.append(d)
                stream.push(
                    {
                        "type": "message_update",
                        "message": {**partial},
                        "delta": {"tool_calls": tc_delta_dicts},
                        "delta_type": "tool_call_delta",
                    }
                )

    except Exception as e:
        # LLM error — build synthetic error message
        error_msg = _build_assistant_message(
            model=config.model,
            stop_reason="aborted" if (signal and signal.is_set()) else "error",
            error_message=str(e),
        )
        if not message_started:
            stream.push({"type": "message_start", "message": error_msg})
        stream.push({"type": "message_end", "message": error_msg})
        context.messages.append(error_msg)
        return error_msg

    # 7. Build finalized assistant message from litellm's stream_chunk_builder
    if not chunks:
        assistant_msg = _build_assistant_message(
            model=config.model,
            stop_reason="aborted" if (signal and signal.is_set()) else "stop",
        )
    else:
        final = litellm.stream_chunk_builder(chunks)
        msg = final.choices[0].message

        # Convert tool_calls from litellm objects to plain dicts
        tool_calls = None
        if msg.tool_calls:
            tool_calls = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                    "provider_specific_fields": getattr(
                        tc, "provider_specific_fields", None
                    ),
                }
                for tc in msg.tool_calls
            ]

        stop_reason = (
            "aborted"
            if (signal and signal.is_set())
            else (chunk_finish_reason or final.choices[0].finish_reason or "stop")
        )

        assistant_msg = _build_assistant_message(
            content=msg.content or None,
            tool_calls=tool_calls,
            thinking_blocks=getattr(msg, "thinking_blocks", None) or None,
            reasoning_content=getattr(msg, "reasoning_content", None) or None,
            provider_specific_fields=getattr(msg, "provider_specific_fields", None),
            model=config.model,
            usage=_extract_usage(final.usage),
            stop_reason=stop_reason,
        )

    # 8. Emit events + append to context
    if not message_started:
        stream.push({"type": "message_start", "message": assistant_msg})
    stream.push({"type": "message_end", "message": assistant_msg})
    context.messages.append(assistant_msg)
    return assistant_msg


# ── Run Loop (the dual while loop engine) ──────────────────────────────────


async def run_loop(context, new_messages, config, signal, stream):
    first_turn = True
    pending = []
    if config.get_steering_messages:
        pending = (await _maybe_await(config.get_steering_messages)) or []

    # OUTER LOOP: follow-ups after agent would stop
    while True:
        has_tool_calls = True
        steering_after_tools = None

        # INNER LOOP: LLM calls + tool execution + steering
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
                    context.messages.append(msg)
                    new_messages.append(msg)
                pending = []

            # Stream assistant response
            assistant_msg = await stream_llm_response(context, config, signal, stream)
            new_messages.append(assistant_msg)

            if assistant_msg["stop_reason"] in ("error", "aborted"):
                stream.push(
                    {"type": "turn_end", "message": assistant_msg, "tool_results": []}
                )
                stream.push({"type": "agent_end", "messages": new_messages})
                stream.end(new_messages)
                return new_messages

            # Check for tool calls
            tool_calls = assistant_msg.get("tool_calls")
            has_tool_calls = bool(tool_calls)

            tool_results = []
            if has_tool_calls:
                execution = await execute_tool_calls(
                    context.tools,
                    assistant_msg,
                    signal,
                    stream,
                    config.get_steering_messages,
                )
                tool_results = execution["tool_results"]
                steering_after_tools = execution.get("steering_messages")

                for tr in tool_results:
                    context.messages.append(tr)
                    new_messages.append(tr)

            stream.push(
                {
                    "type": "turn_end",
                    "message": assistant_msg,
                    "tool_results": tool_results,
                }
            )

            # Get pending for next iteration
            if steering_after_tools:
                pending = steering_after_tools
                steering_after_tools = None
            elif config.get_steering_messages:
                pending = (await _maybe_await(config.get_steering_messages)) or []
            else:
                pending = []

        # Agent would stop — check follow-ups
        if config.get_follow_up_messages:
            follow_ups = (await _maybe_await(config.get_follow_up_messages)) or []
            if follow_ups:
                pending = follow_ups
                continue

        break

    # Normal exit
    stream.push({"type": "agent_end", "messages": new_messages})
    stream.end(new_messages)
    return new_messages


# ── Entry Points ───────────────────────────────────────────────────────────


def agent_loop(prompts, context, config, signal=None):
    """Start an agent loop with new prompt messages. Returns EventStream immediately."""
    stream = EventStream()

    async def _run():
        new_messages = list(prompts)
        try:
            local_ctx = AgentContext(
                system_prompt=context.system_prompt,
                messages=list(context.messages) + list(prompts),
                tools=context.tools,
            )

            stream.push({"type": "agent_start"})
            stream.push({"type": "turn_start"})
            for msg in prompts:
                stream.push({"type": "message_start", "message": msg})
                stream.push({"type": "message_end", "message": msg})

            await run_loop(local_ctx, new_messages, config, signal, stream)
        except Exception as e:
            # Safety net — ensure stream always ends
            error_msg = _build_assistant_message(
                model=config.model,
                stop_reason="aborted" if (signal and signal.is_set()) else "error",
                error_message=str(e),
            )
            stream.push({"type": "agent_end", "messages": [error_msg]})
            stream.end([error_msg])
        finally:
            if not stream._done:
                stream.push({"type": "agent_end", "messages": new_messages})
                stream.end(new_messages)

    asyncio.get_running_loop().create_task(_run())
    return stream


def agent_loop_continue(context, config, signal=None):
    """Continue an agent loop from existing context. Returns EventStream immediately."""
    if not context.messages:
        raise ValueError("Cannot continue: no messages in context")
    if context.messages[-1].get("role") == "assistant":
        raise ValueError("Cannot continue from message role: assistant")

    stream = EventStream()

    async def _run():
        new_messages = []
        try:
            local_ctx = AgentContext(
                system_prompt=context.system_prompt,
                messages=list(context.messages),
                tools=context.tools,
            )
            stream.push({"type": "agent_start"})
            stream.push({"type": "turn_start"})
            await run_loop(local_ctx, new_messages, config, signal, stream)
        except Exception as e:
            error_msg = _build_assistant_message(
                model=config.model,
                stop_reason="aborted" if (signal and signal.is_set()) else "error",
                error_message=str(e),
            )
            stream.push({"type": "agent_end", "messages": [error_msg]})
            stream.end([error_msg])
        finally:
            if not stream._done:
                stream.push({"type": "agent_end", "messages": new_messages})
                stream.end(new_messages)

    asyncio.get_running_loop().create_task(_run())
    return stream
