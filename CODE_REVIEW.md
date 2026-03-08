# Code Review Tracker: liteagent

This file tracks review findings against two baselines:

- `../pi-mono/packages/agent/src/` for intended behavior parity
- `DESIGN_NOTES.md` for documented local differences and intended boundaries

The point of this file is triage, not prose. If a behavior is already documented in
the spec, mark it as a `sharp-edge` or `backlog` item instead of pretending it is
an undiscovered bug.

## Legend

- `open`: likely worth fixing now
- `sharp-edge`: current behavior is documented, but the default is still risky or misleading
- `backlog`: real design debt, not a correctness failure

Priority:

- `P1`: correctness or cancellation gap
- `P2`: target-model fidelity or user-visible behavior gap
- `P3`: maintainability / DX

## Active Summary

| ID | Priority | Status | Summary |
|---|---|---|---|
| `CR-005` | `P3` | `backlog` | Messages are still untyped dicts throughout the core loop |
| `CR-006` | `P3` | `backlog` | Agent and loop both append messages, relying on copied lists |

## Backlog

### `CR-005` Messages are still untyped dicts

- Priority: `P3`
- Status: `backlog`
- Type: maintainability debt vs `pi-mono`
- liteagent:
  - assistant messages are hand-built in multiple places in `liteagent/loop.py` and `liteagent/agent.py`
- pi-mono baseline:
  - typed message unions in `../pi-mono/packages/agent/src/types.ts`
- Why this matters:
  - The code repeatedly constructs near-identical assistant-message dicts with `role`, `content`, `tool_calls`, `thinking_blocks`, `reasoning_content`, `usage`, `stop_reason`, and `timestamp`.
  - A typo or omitted field becomes a silent `dict.get()` failure instead of a type error.
- Suggested fix:
  - Add a shared assistant-message factory first.
  - Consider a lightweight typed message layer after that if the codebase keeps growing.

### `CR-006` Message accumulation is coupled across Agent and loop

- Priority: `P3`
- Status: `backlog`
- Type: design risk
- liteagent:
  - `liteagent/agent.py:328-331`
  - `liteagent/agent.py:407-409`
  - `liteagent/loop.py:497`
  - `liteagent/loop.py:560-561`
- Design notes:
  - `DESIGN_NOTES.md` section `Architecture`
- Why this matters:
  - The loop appends to its local context, and the Agent also appends on every `message_end`.
  - This is safe today because the Agent snapshots message history into a copied list before the run starts.
  - If that copy disappears later, message accumulation will become subtly wrong.
- Why this is not an active bug:
  - The copy is intentional and documented in the spec.
  - So current behavior is correct; the problem is that the ownership boundary is easy to break during refactors.
- Suggested fix:
  - Make ownership explicit in code comments or consolidate message accumulation behind one layer.

## Low-Value Cleanup

These are real but not worth mixing into the active bug list:

- dataclass `__repr__` output is noisy for debugging

## Focused Test Gaps

If adding tests next, do these first:

- (none currently prioritized)
