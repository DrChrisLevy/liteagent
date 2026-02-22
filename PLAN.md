# PLAN.md — How We Build py-pi-agent

## Philosophy

This is a learning project that produces production-quality code. The goal is NOT to ship fast.

1. **Learn** — async Python, agent architectures, streaming, LLM APIs, and everything about pi's agent loop
2. **Have fun** — explore, experiment, go down rabbit holes
3. **Build something real** — a core agent loop as good as pi's, just in Python

## Pair Programming

Code is written together. The agent is an **educational coding expert**, not an autopilot.
The human should understand every line.

**Do NOT:**
- Write the whole thing in one go
- Generate large blocks of code the human hasn't reviewed
- Skip over concepts the human doesn't understand yet

**Do:**
- Work in small steps, from the inside out
- Test everything along the way (unit tests + live API calls)
- Explain WHY, not just WHAT
- Let the human write code too
- Use IPython/REPL for quick experiments

## Building Order

Work from the inside out:

1. **EventStream** (`stream.py`) — the foundation everything flows through
2. **Types** (`types.py`) — Tool, ToolResult, AgentConfig, events
3. **Loop** (`loop.py`) — the dual while loop, streaming, tool execution
4. **Agent** (`agent.py`) — stateful wrapper (prompt, steer, follow_up, abort)
5. **Test runner** — prove it all works together with real API calls

At each step: write the code, write the tests, run it, understand it, then move on.

## Learning Along the Way

When we encounter something the human doesn't understand, we create a **learning deep dive**:

```
learnings/
    async-basics/          # asyncio fundamentals
    streaming-chunks/      # how LLM streaming works
    event-queues/          # asyncio.Queue patterns
    ...
```

Each deep dive is a self-contained exploration — markdown, scripts, notebooks.
Think of them as blog posts the human will later publish.

We **always** go on these detours when needed. Understanding > velocity.

## Project Layout

```
py_pi_agent/               # the library (stream.py, types.py, loop.py, agent.py)
tests/                     # tests + test runner
learnings/                 # deep dive explorations
docs/                      # how the system works (written as we build)
SPEC.md                    # architecture and contracts
PLAN.md                    # this file — how we work
COMPARISONS.md             # framework comparisons
CHANGELOG.md               # what changed and when
AGENTS.md                  # brief instructions for AI agents
```
