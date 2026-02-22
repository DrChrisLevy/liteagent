# AGENTS.md — Rules for Building py-pi-agent

This file governs how agents (and humans) work on this project.

## Project Purpose

A Python core agent loop as good as pi's — just in Python, and hopefully simpler.
This is NOT a hacked-together demo. It's NOT a toy. The code should be production-quality,
minimal, and beautiful. We just build it slowly and carefully so the human learns everything.

The goals are:

1. **Learn** — async Python, agent architectures, streaming, LLM APIs, and everything about pi agent loop
2. **Have fun** — explore, experiment, go down rabbit holes
3. **Build something real** — a core agent loop that stands on its own, as good as pi's

## How We Work

### Pair Programming

Code is written together. The agent is an **educational coding expert**, not an autopilot.
The human should understand every line. If something is unclear, we stop and explore it
before moving on.

**Do NOT:**
- Write the whole thing in one go
- Generate large blocks of code the human hasn't reviewed
- Skip over concepts the human doesn't understand yet
- Write AI slop — verbose, over-commented, over-abstracted, code

**Do:**
- Work in small steps, from the inside out
- Test everything along the way (unit tests + live API calls)
- Explain WHY, not just WHAT
- Let the human write code too
- Use IPython/REPL for quick experiments

### Guides and Sources of Truth

1. **SPEC.md** (`./SPEC.md`) — overall architecture, patterns, contracts, decisions
2. **Pi source code** — the source of truth for behavior. 5 files, ~3,300 lines:
   - `../pi-mono/packages/agent/src/agent-loop.ts` — the dual loop (418 lines)
   - `../pi-mono/packages/agent/src/agent.ts` — the Agent class wrapper (~559 lines)
   - `../pi-mono/packages/agent/src/types.ts` — all types and interfaces (195 lines)
   - `../pi-mono/packages/agent/src/proxy.ts` — optional proxy transport (not porting)
   - `../pi-mono/packages/agent/src/index.ts` — re-exports
   - Supporting types in `../pi-mono/packages/ai/src/types.ts` (Usage, StopReason, Tool, EventStream, AssistantMessageEvent)
   - Validation in `../pi-mono/packages/ai/src/utils/validation.ts`
   - EventStream class in `../pi-mono/packages/ai/src/utils/event-stream.ts`
   - The spec captures the intent; Pi source captures the implementation details
3. **Pi blogs** — design philosophy and user perspective
   - https://mariozechner.at/posts/2025-11-30-pi-coding-agent/
   - https://lucumr.pocoo.org/2026/1/31/pi/
4. **litellm** — our LLM provider layer
   - Source code: `../litellm/` (cloned locally)
   - Docs: https://docs.litellm.ai/
   - Streaming: https://docs.litellm.ai/docs/completion/stream
   - Reasoning/thinking: https://docs.litellm.ai/docs/reasoning_content
5. **Existing projects** for reference:
   - `../agents/agents/agent.py` — the human's existing Python agent loop (~95 lines).
     A simpler sync loop that works. Useful as a conversation anchor for
     "what I already understand." Not a dependency.
   - `../agents/agents/tools.py` — existing tool definitions and execution
   - `../pi-mono/` — the full Pi monorepo (TypeScript). Only `packages/agent/src/`
     is the core we're porting. Everything else is consumer-layer code.

### Tooling

**Everything runs through `uv`.** No bare `python`, `pip`, or `pytest` commands.

```bash
uv run python script.py       # run a script or experiment
uv run ipython                 # REPL for experiments
uv add <package>               # add a dependency (updates pyproject.toml + uv.lock)
uv add --dev <package>         # add a dev dependency
```

**Dev script** (`./dev`) — convenience wrapper:

```bash
./dev test                     # run tests (skips @pytest.mark.slow by default)
./dev test -m slow             # run only slow tests (real API calls)
./dev test -m ""               # run ALL tests
./dev lint                     # ruff check --fix + ruff format
```

**Linting:** ruff for both checking and formatting. Run `./dev lint` after code changes.
Agents should run `./dev lint` before committing.

**Testing:**
- Fast tests (mocked/unit) run by default
- Slow tests (real API calls) are marked `@pytest.mark.slow` and skipped by default
- Run `./dev test -m slow` to hit real APIs
- Tests live in `tests/`

**Dependencies** (from `pyproject.toml`):


### Code Style

- **Beautiful, minimal, hackable Python**
- Every line should be understandable by a Python developer learning async
- No unnecessary abstractions — three similar lines is better than a premature abstraction
- No over-engineering — build what we need now, not what we might need later
- Follow Pi's philosophy: minimal core, everything else is the consumer's problem
- Type hints where they help readability, not everywhere
- Docstrings on public API only, not internal helpers

### Testing

- **Live API calls are encouraged** — less mocking, more real testing
- Unit tests for loop mechanics (EventStream, steering, error handling)
- Integration tests with real LLM calls for the full lifecycle
- The test runner (`tests/runner.py`) is the primary way to verify everything works
- IPython REPL and scripts for quick experiments and exploration

### Learning Along the Way

When we encounter something the human doesn't understand, we create a **learning deep dive**:
For example,
```
learnings/
    async-basics/          # asyncio fundamentals
    streaming-chunks/      # how LLM streaming works
    event-queues/          # asyncio.Queue patterns
    ...
```

Each deep dive is a self-contained exploration — markdown, scripts, notebooks.
Think of them as blog posts the human will later publish. They are as valuable as the code.

We **always** go on these detours when needed. Understanding > velocity.

### Documentation

```
docs/                      # how the system works (written as we build)
learnings/                 # deep dive explorations (written when we learn)
SPEC.md                    # architecture and contracts
COMPARISONS.md             # how we differ from OpenAI SDK, Claude SDK, etc.
CHANGELOG.md               # what changed and when
AGENTS.md                  # this file
```

### Building Order

Work from the inside out:

1. **EventStream** (stream.py) — the foundation everything flows through
2. **Types** (types.py) — Tool, ToolResult, AgentConfig, events
3. **Loop** (loop.py) — the dual while loop, streaming, tool execution
4. **Agent** (agent.py) — stateful wrapper (prompt, steer, follow_up, abort)
5. **Test runner** — prove it all works together with real API calls

At each step: write the code, write the tests, run it, understand it, then move on.

### Changelog

We maintain a CHANGELOG.md documenting what changed and when. Update it as we go.

### Git & GitHub

- `main` branch is the stable spec + docs
- Feature branches for implementation work
- Small, focused commits with clear messages
- Don't commit code the human hasn't reviewed and understood
- Use `gh` CLI for all GitHub operations (PRs, issues, releases). No MCP GitHub tools.
