# AGENTS.md

Python core agent loop inspired by [pi-mono](https://github.com/badlogic/pi-mono) `packages/agent`.
Production-quality, minimal, beautiful Python. As good as pi's — just in Python.

## Build, Test, Lint

Everything runs through `uv`. No bare `python`, `pip`, or `pytest`.

```bash
./dev test                     # run tests (skips @pytest.mark.slow)
./dev test -m slow             # run only slow tests (real API calls)
./dev test -m ""               # run ALL tests
./dev test -m "" -n auto        # run tests in parallel (any combo works with -m, --cov, etc.)
./dev lint                     # ruff check --fix + ruff format
uv run python script.py        # run a script
uv add <package>               # add dependency
```

Run `./dev lint` before committing.

## Creating Tests

- Fast tests by default, slow tests (`@pytest.mark.slow`) need `-m slow`
- **Pure Python logic** (helpers, tool execution, skip logic): fast tests, no mocks
- **Control flow** (steering, follow-ups, error exits): fast tests, thin litellm mock
- **Anything through litellm** (chunks, usage, stop reasons, thinking): live slow tests
- Don't mock what you can test live. Mocks break silently when litellm changes.


## Inspiration for this project

- there are two important projects cloned locally that are used for inspiration and reference:
  - `../litellm/`  - used for the LLM interface.
  - `../pi-mono/` - inspiration for the agent loop.
  - always git pull the latest for these before consulting them.
- This project is a port of pi-mono's agent loop. It is the source of truth for behavior. When in doubt, check the Pi source at `../pi-mono/packages/agent/src/`. However there are some differences in behavior that are documented in `./DESIGN_NOTES.md`. Read `./DESIGN_NOTES.md` for the high-level architecture, intent, and documented differences from pi-mono.

