"""
Stateful agent wrapper around the core loop.

The Agent class manages message history, cancellation signals, event subscription,
and steering/follow-up queues on top of agent_loop/agent_loop_continue.

Faithful port of pi-mono's agent.ts.
"""

import asyncio

from .convert import make_default_convert
from .loop import agent_loop, agent_loop_continue
from .types import AgentConfig, AgentContext, AgentState, _now_ms


def _has_nonempty_str(val):
    """True if val is a non-whitespace string."""
    return isinstance(val, str) and val.strip()


def _any_tool_call_has_name(tool_calls):
    """True if any tool call has a non-empty name (not just empty scaffold)."""
    if not tool_calls:
        return False
    return any((tc.get("function", {}).get("name") or "").strip() for tc in tool_calls)


def _to_user_msg(message):
    """Convert string or dict to a user message dict."""
    if isinstance(message, str):
        return {"role": "user", "content": message, "timestamp": _now_ms()}
    return message


class Agent:
    """Stateful agent that wraps the core loop."""

    def __init__(
        self,
        model,
        tools=None,
        system_prompt="",
        convert_to_llm=None,
        transform_context=None,
        steering_mode="one-at-a-time",
        follow_up_mode="one-at-a-time",
        max_tokens=None,
        temperature=None,
        num_retries=None,
    ):
        self._state = AgentState(
            system_prompt=system_prompt,
            model=model,
            thinking_level="off",
            tools=list(tools) if tools else [],
            messages=[],
            is_streaming=False,
            stream_message=None,
            pending_tool_calls=set(),
            error=None,
        )
        self._convert_to_llm = convert_to_llm or make_default_convert(model)
        self._transform_context = transform_context
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._num_retries = num_retries

        self.steering_mode = steering_mode
        self.follow_up_mode = follow_up_mode

        self._steering_queue: list = []
        self._follow_up_queue: list = []
        self._signal: asyncio.Event | None = None
        self._running_future: asyncio.Future | None = None
        self._subscribers: list = []

    # ── Primary actions (async, same as pi) ───────────────────────────────

    async def prompt(self, message, images=None):
        """Send a message and run the agent loop. Blocks until complete.

        message: str, dict, or list[dict].
        images: optional list of image content blocks (when message is str).
        """
        if self._state.is_streaming:
            raise RuntimeError(
                "Agent is already processing. Use steer() or follow_up() "
                "to queue messages, or await wait_for_idle()."
            )
        if isinstance(message, list):
            prompts = message
        elif isinstance(message, str) and images:
            content = [{"type": "text", "text": message}, *images]
            prompts = [{"role": "user", "content": content, "timestamp": _now_ms()}]
        else:
            prompts = [_to_user_msg(message)]
        await self._run_loop(prompts=prompts)

    async def continue_run(self):
        """Resume from current context."""
        if self._state.is_streaming:
            raise RuntimeError("Agent is already processing.")
        if not self._state.messages:
            raise ValueError("Cannot continue: no messages in context")

        last = self._state.messages[-1]
        if last.get("role") == "assistant":
            if self._steering_queue:
                prompts = self._dequeue_steering()
                await self._run_loop(prompts=prompts, skip_initial_steering_poll=True)
                return
            elif self._follow_up_queue:
                prompts = self._dequeue_follow_ups()
                await self._run_loop(prompts=prompts)
                return
            else:
                raise ValueError(
                    "Cannot continue from assistant message without queued messages. "
                    "Use steer() or follow_up() first."
                )

        await self._run_loop(prompts=None)

    # ── Mid-run message injection ─────────────────────────────────────────

    def steer(self, message):
        """Interrupt agent mid-run. Injected after current tool finishes."""
        self._steering_queue.append(_to_user_msg(message))

    def follow_up(self, message):
        """Queue message for after agent finishes. Deferred delivery."""
        self._follow_up_queue.append(_to_user_msg(message))

    # ── Control ───────────────────────────────────────────────────────────

    def abort(self):
        """Cancel current run."""
        if self._signal:
            self._signal.set()

    async def wait_for_idle(self):
        """Await until agent finishes current run."""
        if self._running_future:
            await self._running_future

    def reset(self):
        """Clear all state. Keep model/tools/system_prompt. Same as pi's reset()."""
        self._state.messages = []
        self._state.error = None
        self._state.is_streaming = False
        self._state.stream_message = None
        self._state.pending_tool_calls = set()
        self._steering_queue.clear()
        self._follow_up_queue.clear()

    # ── Queue management ──────────────────────────────────────────────────

    def clear_steering_queue(self):
        self._steering_queue.clear()

    def clear_follow_up_queue(self):
        self._follow_up_queue.clear()

    def clear_all_queues(self):
        self._steering_queue.clear()
        self._follow_up_queue.clear()

    def has_queued_messages(self):
        return bool(self._steering_queue or self._follow_up_queue)

    def set_steering_mode(self, mode):
        self.steering_mode = mode

    def get_steering_mode(self):
        return self.steering_mode

    def set_follow_up_mode(self, mode):
        self.follow_up_mode = mode

    def get_follow_up_mode(self):
        return self.follow_up_mode

    # ── Message history mutators (same as pi) ─────────────────────────────

    def replace_messages(self, messages):
        self._state.messages = list(messages)

    def append_message(self, message):
        self._state.messages.append(message)

    def clear_messages(self):
        self._state.messages = []

    # ── Configuration (no streaming guard — pi allows mid-run changes) ───

    def set_model(self, model):
        self._state.model = model

    def set_system_prompt(self, prompt):
        self._state.system_prompt = prompt

    def set_tools(self, tools):
        self._state.tools = list(tools) if tools else []

    def set_thinking_level(self, level):
        self._state.thinking_level = level

    # ── Event subscription (primary consumer API) ─────────────────────────

    def subscribe(self, callback):
        """Subscribe to agent events. Returns unsubscribe function."""
        self._subscribers.append(callback)

        def unsubscribe():
            if callback in self._subscribers:
                self._subscribers.remove(callback)

        return unsubscribe

    def _emit(self, event):
        for fn in list(self._subscribers):  # snapshot: safe if fn() unsubscribes
            fn(event)

    # ── State access ──────────────────────────────────────────────────────

    @property
    def state(self):
        return self._state

    @property
    def messages(self):
        return self._state.messages

    # ── Internal: queue dequeuing ─────────────────────────────────────────

    def _dequeue_steering(self):
        if not self._steering_queue:
            return []
        if self.steering_mode == "all":
            msgs = list(self._steering_queue)
            self._steering_queue.clear()
            return msgs
        return [self._steering_queue.pop(0)]

    def _dequeue_follow_ups(self):
        if not self._follow_up_queue:
            return []
        if self.follow_up_mode == "all":
            msgs = list(self._follow_up_queue)
            self._follow_up_queue.clear()
            return msgs
        return [self._follow_up_queue.pop(0)]

    # ── Internal: run loop (sole stream reader, same as pi) ───────────────

    async def _run_loop(self, prompts=None, skip_initial_steering_poll=False):
        """Create signal, build config, call loop, iterate events to update state."""
        self._signal = asyncio.Event()
        self._state.is_streaming = True
        self._state.error = None
        self._running_future = asyncio.get_running_loop().create_future()

        # skipInitialSteeringPoll: when continue_run() already dequeued steering,
        # skip the first poll to avoid double-dequeue. Same as pi line 426-443.
        _skip = skip_initial_steering_poll

        def _steering_hook():
            nonlocal _skip
            if _skip:
                _skip = False
                return []
            return self._dequeue_steering()

        config = AgentConfig(
            model=self._state.model,
            convert_to_llm=self._convert_to_llm,
            transform_context=self._transform_context,
            get_steering_messages=_steering_hook,
            get_follow_up_messages=self._dequeue_follow_ups,
            reasoning_effort=(
                None
                if self._state.thinking_level == "off"
                else self._state.thinking_level
            ),
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            num_retries=self._num_retries,
        )

        context = AgentContext(
            system_prompt=self._state.system_prompt,
            messages=list(self._state.messages),
            tools=list(self._state.tools) if self._state.tools else None,
        )

        try:
            if prompts is not None:
                stream = agent_loop(prompts, context, config, self._signal)
            else:
                stream = agent_loop_continue(context, config, self._signal)

            # Sole reader — iterate events, update state, emit to subscribers
            async for event in stream:
                self._handle_event(event)
                self._emit(event)

            # Post-stream partial handling (pi agent.ts line 504-518):
            # If aborted mid-stream, a partial assistant message may remain.
            # Append it if it has real content; raise if only empty scaffolding.
            partial = self._state.stream_message
            if partial and partial.get("role") == "assistant":
                content = partial.get("content")
                # Note: thinking_blocks are NOT accumulated on the partial during
                # streaming (loop.py emits them as deltas only). Anthropic thinking
                # is also surfaced as reasoning_content by litellm, so the
                # reasoning_content check covers the thinking-only abort case.
                has_real_content = (
                    (isinstance(content, str) and content.strip())
                    or _any_tool_call_has_name(partial.get("tool_calls"))
                    or _has_nonempty_str(partial.get("reasoning_content"))
                )
                if has_real_content:
                    self._state.messages.append(partial)
                elif self._signal and self._signal.is_set():
                    raise Exception("Request was aborted")

        except Exception as e:
            error_msg = {
                "role": "assistant",
                "content": None,
                "tool_calls": None,
                "thinking_blocks": None,
                "reasoning_content": None,
                "model": self._state.model,
                "stop_reason": "aborted"
                if (self._signal and self._signal.is_set())
                else "error",
                "error_message": str(e),
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "cache_read_tokens": 0,
                    "cache_creation_tokens": 0,
                },
                "timestamp": _now_ms(),
            }
            self._state.messages.append(error_msg)
            self._state.error = str(e)
            self._emit({"type": "agent_end", "messages": [error_msg]})

        finally:
            self._state.is_streaming = False
            self._state.stream_message = None
            self._state.pending_tool_calls = set()
            self._signal = None
            if self._running_future and not self._running_future.done():
                self._running_future.set_result(None)
            self._running_future = None

    def _handle_event(self, event):
        """Update AgentState from events — same as pi's event loop in _runLoop."""
        t = event["type"]
        if t == "message_start":
            if event["message"].get("role") == "assistant":
                self._state.stream_message = event["message"]
        elif t == "message_update":
            self._state.stream_message = event["message"]
        elif t == "message_end":
            self._state.stream_message = None
            self._state.messages.append(event["message"])
        elif t == "tool_execution_start":
            self._state.pending_tool_calls.add(event["tool_call_id"])
        elif t == "tool_execution_end":
            self._state.pending_tool_calls.discard(event["tool_call_id"])
        elif t == "turn_end":
            msg = event.get("message", {})
            if msg.get("error_message"):
                self._state.error = msg["error_message"]
        elif t == "agent_end":
            self._state.is_streaming = False
