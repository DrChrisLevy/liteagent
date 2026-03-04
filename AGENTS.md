# AGENTS.md

Python core agent loop inspired by [pi-mono](https://github.com/badlogic/pi-mono) `packages/agent`.
Production-quality, minimal, beautiful Python. As good as pi's — just in Python.

## Build, Test, Lint

Everything runs through `uv`. No bare `python`, `pip`, or `pytest`.

```bash
./dev test                     # run tests (skips @pytest.mark.slow)
./dev test -m slow             # run only slow tests (real API calls)
./dev test -m ""               # run ALL tests
./dev lint                     # ruff check --fix + ruff format
uv run python script.py        # run a script
uv add <package>               # add dependency
```

Run `./dev lint` before committing.

## Code Style

- Beautiful, minimal, hackable Python
- No unnecessary abstractions — three similar lines > premature abstraction
- No over-engineering — build what we need now
- No AI slop — verbose, over-commented, over-abstracted code
- Type hints where they help readability, not everywhere
- Docstrings on public API only

## Key Files

### Sources of truth
- **`./SPEC.md`** — read this first. Architecture, contracts, event types, tool protocol, everything.
- `../pi-mono/packages/agent/src/` — Pi core (5 files, ~3,300 lines). **The source of truth for behavior.** When in doubt, check the Pi source.
  - `agent-loop.ts` — the dual loop
  - `agent.ts` — the Agent class wrapper
  - `types.ts` — all types and interfaces
  - Supporting: `../pi-mono/packages/ai/src/types.ts`, `../pi-mono/packages/ai/src/utils/event-stream.ts`, `../pi-mono/packages/ai/src/utils/validation.ts`

### Reference
- `../litellm/` — litellm source (cloned locally). Docs: https://docs.litellm.ai/
- `../agents/agents/agent.py` — human's existing Python agent loop (~95 lines)
- Pi blogs: https://mariozechner.at/posts/2025-11-30-pi-coding-agent/ and https://lucumr.pocoo.org/2026/1/31/pi/

### Project docs
- `PLAN.md` — how we work, build order, learning philosophy
- `COMPARISONS.md` — how we differ from OpenAI SDK, Claude SDK, Pydantic AI, pi-mono (includes fidelity scorecard + litellm gap analysis)

## Git & GitHub

- `main` branch is stable. Feature branches for implementation.
- Small, focused commits.
- Use `gh` CLI for all GitHub operations. No MCP GitHub tools.

## Testing

- Fast tests by default, slow tests (`@pytest.mark.slow`) need `-m slow`
- **Pure Python logic** (helpers, tool execution, skip logic): fast tests, no mocks
- **Control flow** (steering, follow-ups, error exits): fast tests, thin litellm mock
- **Anything through litellm** (chunks, usage, stop reasons, thinking): live slow tests
- Don't mock what you can test live. Mocks break silently when litellm changes.
- Target models (must pass all five, four currently active):
  - `anthropic/claude-opus-4-6`
  - `anthropic/claude-sonnet-4-6`
  - `gemini/gemini-3-flash-preview`
  - `gemini/gemini-3.1-pro-preview` *(disabled — consistently timing out since March 2026)*
  - `gpt-5.2`
