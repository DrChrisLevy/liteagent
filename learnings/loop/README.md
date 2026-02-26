# Learning: The Dual Loop (from scratch)


## Part 1: What problem are we solving?

An LLM by itself is stateless — you send it a prompt, it sends back text. That's
one round trip. But "agents" need multiple rounds:

```
You: "Delete temp files"
LLM: [calls bash tool: rm -rf /tmp/old_*]
     "Done. Deleted 47 files."
You: "Now check disk space"
LLM: [calls bash tool: df -h]
     "You have 120GB free on /."
```

Each "LLM call → tool execution → feed result back" is a **turn**. The loop's
job is automating these turns: call the LLM, check if it wants tools, execute
them, feed results back, repeat until the LLM says "stop."

But there's a twist: users can interrupt. Mid-tool-execution, a user might say
"actually, do this instead." And after the agent finishes, they might have
follow-up messages queued. This gives us **two loops**.


## Part 2: The dual loop

### Inner loop: LLM + tools + steering

```
while has_tool_calls or pending_messages:
    inject any pending messages into conversation
    call LLM (streaming)
    if error/aborted: exit
    if LLM wants tools: execute them (checking for steering after each)
    check for new steering messages
```

The inner loop handles the core cycle. It keeps running as long as:
- The LLM wants to call tools (`has_tool_calls`), OR
- There are user steering messages to inject (`pending_messages`)

**Steering** is immediate interruption. When a user types while the agent is
executing tools, their message goes into a steering queue. After each tool
completes, the loop checks this queue. If messages are waiting:
1. Skip remaining tools (with synthetic "skipped" results)
2. Inject the steering messages into the conversation
3. Call the LLM again (it sees the new user message)

### Outer loop: follow-ups

```
while True:
    [run inner loop until it settles]

    follow_ups = check_follow_up_queue()
    if follow_ups:
        pending_messages = follow_ups
        continue   # back to inner loop

    break  # truly done
```

After the inner loop finishes (no more tool calls, no steering), the agent
would normally stop. But the outer loop checks one more queue: **follow-ups**.
Follow-ups are messages queued with "when you're done, also do this." If any
exist, they become `pending_messages` and the inner loop runs again.

### Steering vs follow-ups

| | Steering | Follow-up |
|---|---|---|
| **When** | During tool execution | After agent settles |
| **Effect** | Interrupts, skips remaining tools | Continues with new context |
| **Analogy** | "Stop! Do this instead" | "Also, when you're done..." |

### The firstTurn flag

The entry point (`agent_loop`) emits `turn_start` before calling `run_loop`.
So the first iteration of the inner loop must NOT emit `turn_start` again.
The `first_turn` flag handles this — it starts `True`, and on the first
iteration we skip the event and set it to `False`.

```python
if not first_turn:
    stream.push({"type": "turn_start"})
else:
    first_turn = False
```


## Part 3: Streaming

### Why stream?

Without streaming, the user sees nothing until the entire response is done.
With streaming, tokens appear word-by-word as the LLM generates them.

litellm's `acompletion(stream=True)` returns an async iterator of chunks.
Each chunk has a `delta` with partial content:

```
chunk 1: delta.content = "The"
chunk 2: delta.content = " answer"
chunk 3: delta.content = " is"
chunk 4: delta.content = " 42."
chunk 5: finish_reason = "stop"  (final chunk, may include usage)
```

### Chunk anatomy

```python
chunk.choices[0].delta.content           # text token (str or None)
chunk.choices[0].delta.reasoning_content # thinking token (str or None)
chunk.choices[0].delta.thinking_blocks   # Anthropic thinking blocks (list or None)
chunk.choices[0].delta.tool_calls        # tool call fragments (list or None)
chunk.choices[0].finish_reason           # None during streaming, set on final
chunk.usage                              # None unless stream_options include_usage
```

**Important:** Not all attributes exist on every chunk. litellm deletes None
attributes rather than setting them to None. Always use `getattr(delta, "attr", None)`.

### Delta accumulation

We accumulate four types of content during streaming:

**Text** — simple concatenation:
```python
content_parts = []
for chunk in response:
    if delta_content:
        content_parts.append(delta_content)
final_text = "".join(content_parts)
```

**Reasoning** — same pattern:
```python
reasoning_parts = []
if delta_rc:
    reasoning_parts.append(delta_rc)
```

**Tool calls** — merge by index. litellm streams tool calls as indexed fragments.
The first chunk for a tool call has the `id` and `name`. Subsequent chunks
have partial `arguments` JSON:

```
chunk 1: index=0, id="call_123", function.name="bash", function.arguments=""
chunk 2: index=0, id=None, function.name=None, function.arguments='{"comm'
chunk 3: index=0, id=None, function.name=None, function.arguments='and": "ls"}'
```

We accumulate by index in a dict, concatenating the arguments string:

```python
tool_calls_acc = {}  # index -> {"id", "function": {"name", "arguments"}}
for tc_delta in delta_tc:
    idx = tc_delta.index
    if idx not in tool_calls_acc:
        tool_calls_acc[idx] = {"id": "", "type": "function", "function": {"name": "", "arguments": ""}}
    if tc_delta.id:
        tool_calls_acc[idx]["id"] = tc_delta.id
    if tc_delta.function:
        if tc_delta.function.name:
            tool_calls_acc[idx]["function"]["name"] += tc_delta.function.name
        if tc_delta.function.arguments:
            tool_calls_acc[idx]["function"]["arguments"] += tc_delta.function.arguments
```

### The thinking block gotcha

**This is the trickiest part of streaming.**

Anthropic returns `thinking_blocks` — partial fragments of internal reasoning
that include cryptographic signatures. The catch: each chunk has a fragment
with partial text and an **empty signature**. Only the final chunk for a block
carries the real signature.

```
chunk 1: thinking_blocks = [{"type": "thinking", "thinking": "Let me", "signature": ""}]
chunk 2: thinking_blocks = [{"type": "thinking", "thinking": " think", "signature": ""}]
chunk 3: thinking_blocks = [{"type": "thinking", "thinking": "...",    "signature": "sig_abc"}]
```

**WRONG** — extending raw deltas creates many small blocks. Anthropic rejects them:
```python
thinking_blocks.extend(delta_blocks)  # DON'T DO THIS
```

**RIGHT** — merge fragments, finalize when signature arrives:
```python
thinking_cur = None  # current block being accumulated

for b in delta_blocks:
    if b.get("type") == "thinking":
        if thinking_cur is None:
            thinking_cur = {"type": "thinking", "thinking": "", "signature": ""}
        thinking_cur["thinking"] += b.get("thinking", "")
        sig = b.get("signature", "")
        if sig:
            thinking_cur["signature"] = sig
            thinking_blocks.append(thinking_cur)  # finalized!
            thinking_cur = None
```


## Part 4: Tool execution

### Sequential, not parallel

Tools execute one at a time, in order. This is a deliberate design choice
(same as Pi). Why?

1. Tools may have side effects (file writes, API calls)
2. Later tools may depend on earlier ones
3. Steering needs a clean interleaving point (check after each tool)

### The error-as-result pattern

When a tool fails, the loop does NOT crash. Instead, it wraps the error as a
tool result with `is_error=True` and feeds it back to the LLM. The LLM sees
the error text and can decide what to do (retry, try differently, report to user).

```python
try:
    result = await tool.execute(tc_id, validated_args, signal, on_update)
    is_error = False
except Exception as e:
    result = ToolResult(content=[{"type": "text", "text": str(e)}])
    is_error = True
```

This applies to three failure modes:
- **Tool not found**: LLM hallucinated a tool name
- **Validation error**: Pydantic rejected the arguments
- **Execution error**: The tool itself threw an exception

All three produce error tool results. The loop continues. Only LLM errors
(network failure, rate limit) stop the loop.

### Steering interrupts

After each tool completes, the loop checks for steering messages:

```python
if get_steering_messages:
    steering = await get_steering_messages()
    if steering:
        # Skip remaining tools
        for skipped in tool_calls[i + 1:]:
            results.append(_skip_tool_call(skipped, stream))
        break
```

Skipped tools get synthetic results: `"Skipped due to queued user message."` with
`is_error=True`. This keeps the tool call/result pairing balanced — some
providers reject orphaned tool calls without matching results.


## Part 5: Entry points

### `agent_loop(prompts, context, config, signal)`

Starts a new agent run. The prompts are user messages to add to the conversation.

```python
def agent_loop(prompts, context, config, signal=None):
    stream = EventStream()     # consumer will iterate this

    async def _run():
        context.messages.extend(prompts)
        stream.push({"type": "agent_start"})
        stream.push({"type": "turn_start"})
        for msg in prompts:
            stream.push({"type": "message_start", "message": msg})
            stream.push({"type": "message_end", "message": msg})
        await run_loop(context, new_messages, config, signal, stream)

    asyncio.get_running_loop().create_task(_run())
    return stream  # returns IMMEDIATELY
```

Key insight: `agent_loop` returns the stream **before** any LLM work starts.
The actual work runs in a background `asyncio.Task`. The consumer iterates the
stream and sees events as they arrive.

### `agent_loop_continue(context, config, signal)`

Resumes from existing context — used for retries after errors. Same pattern,
but no new prompts. Validates that context has messages and the last message
isn't from the assistant (the LLM needs something to respond to).


## Part 6: Event flow

Complete event sequence for a turn with one tool call:

```
agent_start                          # loop started
turn_start                           # first turn begins
message_start  (user prompt)         # user message injected
message_end    (user prompt)
message_start  (assistant)           # LLM starts streaming
message_update (text_delta)          # "Let me check"
message_update (text_delta)          # " that for you"
message_update (tool_call_delta)     # tool call building
message_update (tool_call_delta)     # arguments accumulating
message_end    (assistant)           # LLM finished, tool_calls present
tool_execution_start                 # tool begins
tool_execution_update                # partial result (if tool streams)
tool_execution_end                   # tool finished
message_start  (tool result)         # tool result added
message_end    (tool result)
turn_end                             # turn complete (assistant msg + tool results)
turn_start                           # next turn (tools were called, loop continues)
message_start  (assistant)           # LLM responds to tool result
message_update (text_delta)          # "The answer is 42."
message_end    (assistant)           # stop_reason="stop"
turn_end                             # final turn complete
agent_end                            # loop finished, contains all new_messages
```


## Part 7: Error handling summary

| Error type | Behavior | Stop reason |
|---|---|---|
| Tool not found | Error result to LLM, loop continues | n/a |
| Pydantic validation | Error result to LLM, loop continues | n/a |
| Tool throws | Error result to LLM, loop continues | n/a |
| LLM network error | Synthetic error message, loop stops | `"error"` |
| LLM rate limit | Synthetic error message, loop stops | `"error"` |
| Signal abort | Break streaming, loop stops | `"aborted"` |
| Unhandled exception | Safety net in entry point, stream ends | `"error"` |

After an error, the consumer can call `agent.continue_run()` to retry from the
same context. No automatic retry — the consumer decides.


## Part 8: `litellm.modify_params = True`

This global flag is set at module import. It tells litellm to auto-fix common
message format issues before sending to providers:

- **Alternating roles**: Anthropic requires user/assistant alternation. If we
  have consecutive user messages (e.g., steering injection), litellm inserts
  placeholder messages.
- **Orphaned tool calls**: If messages contain tool_call blocks but no `tools=`
  parameter, litellm adds a dummy tool definition.
- **Empty thinking**: If assistant messages lack `thinking_blocks` but the
  thinking parameter is set, litellm drops it.
- **Missing tool results**: Sanitizes orphaned tool calls without matching results.

Without this, our loop would need to handle all these provider-specific
constraints manually. With it, we can focus on the core logic and let litellm
fix the edges.
