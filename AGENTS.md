# AGENTS.md

Python core agent loop inspired by [pi-mono](https://github.com/badlogic/pi-mono) `packages/agent`.
Production-quality, minimal, beautiful Python. As good as pi's — just in Python.

## Build, Test, Lint

Everything runs through `uv`. No bare `python`, `pip`, or `pytest`.

```bash
./dev test                     # run tests (skips @pytest.mark.slow)
./dev test -m slow             # run only slow tests (real API calls)
./dev test -m ""               # run ALL tests
./dev test -n auto             # run tests in parallel (any combo works with -m, --cov, etc.)
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

## Inspiration for this project

- This project is a port of pi-mono's agent loop. It is the source of truth for behavior. When in doubt, check the Pi source. It can be found at `../pi-mono/packages/agent/src/`. There are some minor differences in the implementation, but the behavior is the same for the most part.
- See learnings/ for more details on documentation and differences.
- Pi blogs: https://mariozechner.at/posts/2025-11-30-pi-coding-agent/ and https://lucumr.pocoo.org/2026/1/31/pi/
  - explains some philosophy and design decisions.
- `../litellm/` — litellm source (cloned locally). Docs: https://docs.litellm.ai/
  - we use litellm to interface with the LLM.

## Testing

- Fast tests by default, slow tests (`@pytest.mark.slow`) need `-m slow`
- **Pure Python logic** (helpers, tool execution, skip logic): fast tests, no mocks
- **Control flow** (steering, follow-ups, error exits): fast tests, thin litellm mock
- **Anything through litellm** (chunks, usage, stop reasons, thinking): live slow tests
- Don't mock what you can test live. Mocks break silently when litellm changes.
