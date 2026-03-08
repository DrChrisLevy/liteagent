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
| `CR-005` | `P3` | `backlog` | Messages are plain dicts — no typed layer yet |

## Backlog

### `CR-005` Messages are plain dicts — no typed layer yet

- Priority: `P3`
- Status: `backlog`
- Type: maintainability debt vs `pi-mono`
- What we did:
  - Added `_build_assistant_message` and `_build_tool_result_message` factories in `types.py`
  - All construction sites now go through the factories (consistent shape, one place to update)
- What remains:
  - Messages are still plain dicts — readers use `.get()` / `[]` with no type checking
  - pi-mono has typed unions (`UserMessage | AssistantMessage | ToolResultMessage`)
  - A `TypedDict` layer would catch field typos at the read sites too
  - Not urgent — the factories already prevent construction bugs


## Low-Value Cleanup

These are real but not worth mixing into the active bug list:

- dataclass `__repr__` output is noisy for debugging

## Focused Test Gaps

If adding tests next, do these first:

- (none currently prioritized)
