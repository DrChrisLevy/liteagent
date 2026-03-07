# Code Review: liteagent

## What I'd fix

### 1. Messages are untyped dicts — this is the project's biggest weakness

Every message in the system is a raw `dict`. Assistant messages are constructed in at least 6 places:

- `loop.py:423-434` (error in stream_llm_response)
- `loop.py:443-452` (empty chunks fallback)
- `loop.py:481-491` (normal assistant message)
- `loop.py:620-632` (agent_loop safety net)
- `loop.py:663-675` (agent_loop_continue safety net)
- `agent.py:366-385` (Agent._run_loop error handler)

Each is a ~10-key dict literal with `role`, `content`, `tool_calls`, `thinking_blocks`, `reasoning_content`, `usage`, `stop_reason`, `timestamp`, etc. They're almost identical but not quite. One typo — `"reasoning_contnet"` — and you get a silent bug that passes all tests because `dict.get()` just returns `None`.

Pi-mono avoids this with TypeScript types (`AssistantMessage`, `ToolResultMessage`). You already have dataclasses in `types.py`. Use them. A `_make_assistant_msg()` factory or an `AssistantMessage` dataclass would eliminate an entire class of bugs.

### 2. `_default_convert_to_llm` silently drops images from tool results

```python
text_parts = [b["text"] for b in content if b.get("type") == "text"]
content = "\n".join(text_parts)
```

Any `image_url` blocks in tool results are silently discarded. The test suite's `make_convert_to_llm` (in `test_loop.py`) handles images correctly — it even handles the OpenAI-specific workaround of shuffling images into a synthetic user message. But the default converter shipped with the library doesn't.

A user doing `Agent(model="anthropic/claude-sonnet-4-6")` with a tool that returns images will lose those images silently. This should either handle images, or at minimum log a warning.

### 3. `_default_convert_to_llm` also drops `provider_specific_fields`

The default converter preserves `thinking_blocks` and `reasoning_content` on
assistant messages, but it strips `provider_specific_fields`.

That matters for Gemini. litellm stores Gemini `thought_signatures` inside
`provider_specific_fields`, and those signatures are part of the multi-turn
thinking fidelity story documented elsewhere in the repo. The raw loop preserves
them. The default `Agent` path drops them unless the caller knows to supply a
custom `convert_to_llm`.

The current slow tests prove that `provider_specific_fields` survive through the
loop, but they do not prove the default Agent converter preserves them.

### 4. Abort cannot interrupt a slow `litellm.acompletion()` await

`Agent.abort()` sets an `asyncio.Event`, and `stream_llm_response()` checks that
signal while iterating chunks. But it does **not** check the signal until after:

```python
response = await litellm.acompletion(**kwargs)
```

That means abort only works once the provider has already returned a streaming
response object. If the provider call itself is slow, rate-limited, or hung,
the agent stays busy even though the user already aborted.

The tests around abort are good, but the coverage currently stops short of this
specific edge: "abort before first chunk arrives, while `acompletion()` itself is
still pending." That's the real hole.

### 5. `_extract_usage()` misses nested `cache_creation_tokens`

`_extract_usage()` reads:

- Anthropic-style top-level `cache_creation_input_tokens`
- OpenAI-style nested `prompt_tokens_details.cached_tokens`

But it does **not** read nested
`usage.prompt_tokens_details.cache_creation_tokens`, which litellm also exposes.
So cache-write accounting is incomplete for providers that use the nested shape.

This is not catastrophic, but it does mean usage metadata can under-report
cache creation even when litellm has the data.

### 6. `_now_ms()` is duplicated

Defined identically in `loop.py:33` and `agent.py:17`. It's one line (`int(time.time() * 1000)`) so it's not a big deal, but it's the kind of thing that signals "these files were written separately and never cleaned up." Put it in a shared spot or just inline it.

### 7. `_dequeue_steering` / `_dequeue_follow_ups` use `list.pop(0)`

```python
return [self._steering_queue.pop(0)]
```

`pop(0)` on a Python list is O(n) — it shifts every element. Use `collections.deque` with `popleft()`. In practice, steering queues are tiny so this doesn't matter. But it's a code smell.

### 8. `_handle_event` double-appends messages

The Agent's `_handle_event` does `self._state.messages.append(event["message"])` on every `message_end` event. But `run_loop` already appends to `context.messages` inside the loop. These are different lists (because of the `list()` copy at `agent.py:330`), so it works — but only because of that shallow copy. The Agent and the loop each think they own message accumulation. This is fragile coupling. If someone removes the `list()` copy, messages get double-appended.

### 9. No `__repr__` on dataclasses

`AgentState` will dump the entire message history when printed. `Tool` will dump its full JSON schema. `AgentConfig` will dump the convert_to_llm function object. This makes debugging painful. Even a simple `__repr__` that shows just the class name and key fields would help.

### 10. The mock infrastructure is duplicated across test files

`_Obj`, `make_delta`, `make_chunk`, `make_final`, `async_iter`, `mock_llm`, `mock_llm_seq` are copy-pasted between `test_loop.py` and `test_agent.py`. They're identical. Extract to a `tests/conftest.py` or `tests/helpers.py`.

---

## Architectural fidelity to pi-mono

The port is faithful. The important stuff is all there:

- Dual while loop (inner: tool calls + steering, outer: follow-ups) — matches pi's `runLoop`
- Sequential tool execution with steering checks after each tool
- `skipToolCall` pattern for remaining tools when steering arrives
- `skipInitialSteeringPoll` to avoid double-dequeue in `continue_run` — matches pi's `agent.ts:426-443`
- `agent_loop` vs `agent_loop_continue` entry points — matches pi's `agentLoop` / `agentLoopContinue`
- Agent state management (stream_message, pending_tool_calls, is_streaming, error) tracks events identically
- Partial preservation on abort — checks for real content vs empty scaffold, same as pi's `agent.ts:504-518`

The divergences are deliberate and well-reasoned:
- litellm replaces pi-ai's entire streaming/provider layer
- `asyncio.Event` replaces `AbortController` (correct Python equivalent)
- `asyncio.Future` replaces Promise-based `runningPrompt`
- Pydantic replaces AJV for tool argument validation
- `convert_to_llm` takes plain dicts instead of typed messages (consequence of the untyped-messages approach)

---

## Is it AI slop?

**No.** Several things tell me a human with real understanding wrote this:

1. **The litellm bug workarounds are real.** The `chunk_finish_reason` capture, the `xfail` tests with specific bug descriptions, the `litellm.modify_params = True` line — these come from actually hitting problems, not from reading docs.

2. **The `_make_on_update` closure factory** in `execute_tool_calls` (loop.py:180-195) exists because of a genuine Python closure bug — loop variables captured by reference, not by value. An AI would either miss this or over-explain it.

3. **The thinking blocks dedup logic** (`if not delta_rc:` guard at loop.py:355) exists because Anthropic sometimes sends both `reasoning_content` and `thinking_blocks` on the same chunk. You only know this from testing against the real API.

4. **The slow tests are not synthetic.** The multimodal spike test generates a real chart with matplotlib, uses randomized spike placement, and verifies the LLM can identify it. The Gemini `provider_specific_fields` test checks for `thought_signatures` — you'd only know about these from actually working with Gemini thinking mode.

5. **The commit history tells a story.** EventStream first, then types, then the loop (as a PR), then the Agent class, then targeted fixes for divergences found by comparison. That's how someone builds a thing incrementally, not how someone generates a complete project in one shot.

Two mild tells:
- The section dividers (`# ── Group 5: run_loop ──`) with box-drawing characters. Some developers use these genuinely, but they're more common in AI-generated code. Ambiguous.
- The `COMPARISONS.md` "fidelity scorecard" with exact percentages (99% loop, 98% tools, 95% events, etc.) feels like something an AI would generate when asked "how faithful is this port?" A human would say "very close" or "a few gaps in streaming."

---

## Test gaps

- **Concurrent access**: What if `steer()` is called from one coroutine while `prompt()` is running in another? The steering queue is a plain list. Python async can interleave at `await` points. Probably fine, but untested.
- **`_default_convert_to_llm`** with images: tested in isolation for thinking blocks, but no test proves that images actually get dropped.
- **`_default_convert_to_llm` with Gemini metadata**: no test proves the default Agent converter preserves `provider_specific_fields` / `thought_signatures`.
- **Abort before first chunk**: no test proves a pending `litellm.acompletion()` can actually be interrupted before it yields a response object.
- **Nested cache creation accounting**: no direct test covers `usage.prompt_tokens_details.cache_creation_tokens`.
- **No test for `agent_loop_continue` with a pre-existing multi-turn history** — the existing test starts from a tool result, but doesn't test resuming after multiple conversation turns with compacted context.

---
