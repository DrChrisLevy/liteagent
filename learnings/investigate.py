"""
Raw litellm investigation script.

4-turn multimodal scenario, identical for every model:
  Turn 1: User sends image + text → model should call tool
  Turn 2: Tool returns image + text → model responds
  Turn 3: User sends second image + follow-up → model responds
  Turn 4: User asks about all images → model responds

Every response is dumped raw. Nothing hidden. No provider-specific branches.
No assumptions about which fields exist — dumps everything litellm returns.

Usage:
    uv run python investigate.py                    # run all models
    uv run python investigate.py sonnet             # run matching models
    uv run python investigate.py gemini flash       # multiple filters (OR)
"""

import asyncio
import base64
import io
import sys
import time
import traceback

import litellm
import numpy as np

litellm.modify_params = True

# ── Models ────────────────────────────────────────────────────────────────

MODELS = [
    {"model": "anthropic/claude-sonnet-4-6", "reasoning_effort": "high"},
    {"model": "anthropic/claude-opus-4-6", "reasoning_effort": "high"},
    {"model": "gemini/gemini-3-pro-preview", "reasoning_effort": "high"},
    {"model": "gemini/gemini-3-flash-preview", "reasoning_effort": "high"},
    {"model": "gemini/gemini-3.1-pro-preview", "reasoning_effort": "high"},
    {"model": "gpt-5.2", "reasoning_effort": "high"},
    {"model": "gpt-5.3-codex", "reasoning_effort": "high"},
    {"model": "gpt-5.4"},
]

# ── Image generation ─────────────────────────────────────────────────────


def _setup_mpl():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    import matplotlib.pyplot as plt

    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def make_bar_chart():
    plt = _setup_mpl()
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun"]
    values = [120, 135, 128, 142, 580, 131]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(months, values, color=["#3498db" if v < 300 else "#e74c3c" for v in values])
    ax.set_title("Monthly API Errors (2026 H1)")
    ax.set_ylabel("Error Count")
    for i, v in enumerate(values):
        ax.text(i, v + 15, str(v), ha="center", fontweight="bold")
    plt.tight_layout()
    return _to_b64(fig)


def make_scatter_plot():
    plt = _setup_mpl()
    np.random.seed(42)
    x = np.append(np.random.normal(50, 10, 200), [90, 92, 88, 91, 93])
    y = np.append(
        2.3 * np.random.normal(50, 10, 200) + np.random.normal(0, 15, 200),
        [50, 55, 48, 52, 47],
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(x[:200], y[:200], alpha=0.5, c="#3498db", label="Normal")
    ax.scatter(x[200:], y[200:], alpha=0.8, c="#e74c3c", s=80, label="Outlier cluster")
    ax.set_xlabel("Request Latency (ms)")
    ax.set_ylabel("Memory Usage (MB)")
    ax.set_title("Latency vs Memory — Outlier Detection")
    ax.legend()
    plt.tight_layout()
    return _to_b64(fig)


def make_heatmap():
    plt = _setup_mpl()
    np.random.seed(99)
    data = np.random.poisson(5, (7, 24))
    data[2, 2:5] = [45, 52, 38]  # Wednesday 2-4am hotspot
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.imshow(data, cmap="YlOrRd", aspect="auto")
    ax.set_yticks(range(7), ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    ax.set_xlabel("Hour of Day")
    ax.set_title("Error Rate Heatmap — May 2026")
    plt.tight_layout()
    return _to_b64(fig)


# ── Tool definitions ─────────────────────────────────────────────────────

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "analyze_metrics",
            "description": "Run statistical analysis on server metrics. Returns a text summary and a scatter plot.",
            "parameters": {
                "type": "object",
                "properties": {
                    "metric_name": {
                        "type": "string",
                        "description": "Name of the metric to analyze",
                    },
                    "time_range": {
                        "type": "string",
                        "description": "Time range (e.g. '7d', '30d')",
                    },
                },
                "required": ["metric_name"],
            },
        },
    }
]


# ── Helpers ───────────────────────────────────────────────────────────────


def img(b64):
    return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}


def txt(text):
    return {"type": "text", "text": text}


def dump(obj, indent=0):
    """Recursively dump ANY object. Includes private attrs. Hides nothing. No truncation."""
    prefix = "  " * indent
    if obj is None:
        return f"{prefix}None"
    if isinstance(obj, (int, float, bool)):
        return f"{prefix}{repr(obj)}"
    if isinstance(obj, str):
        return f"{prefix}{repr(obj)}"
    if isinstance(obj, list):
        if not obj:
            return f"{prefix}[]"
        lines = [f"{prefix}["]
        for item in obj:
            lines.append(dump(item, indent + 1) + ",")
        lines.append(f"{prefix}]")
        return "\n".join(lines)
    if isinstance(obj, dict):
        if not obj:
            return f"{prefix}{{}}"
        lines = [f"{prefix}{{"]
        for k, v in obj.items():
            lines.append(f"{prefix}  {k!r}: {dump(v, indent + 1).lstrip()},")
        lines.append(f"{prefix}}}")
        return "\n".join(lines)
    if isinstance(obj, set):
        return f"{prefix}{repr(obj)}"
    # Object with attributes — dump ALL of them, including private
    if hasattr(obj, "__dict__"):
        all_attrs = {}
        # Get all attributes from __dict__ (instance attrs)
        for k, v in vars(obj).items():
            all_attrs[k] = v
        # Also try common litellm attributes that might be properties/slots
        for attr in ["model_extra", "model_fields_set"]:
            if hasattr(obj, attr) and attr not in all_attrs:
                try:
                    all_attrs[attr] = getattr(obj, attr)
                except Exception:
                    pass
        lines = [f"{prefix}<{type(obj).__name__}>"]
        for k, v in sorted(all_attrs.items()):
            v_str = dump(v, indent + 1).lstrip()
            lines.append(f"{prefix}  .{k} = {v_str}")
        return "\n".join(lines)
    return f"{prefix}{repr(obj)}"


def dump_all_attrs(obj, label=""):
    """Print every attribute on an object — public, private, properties. Nothing hidden."""
    if label:
        print(f"\n  {label}:")
    if obj is None:
        print("    None")
        return

    for k, v in sorted(vars(obj).items()):
        print(f"    .{k} = {dump(v, 2).lstrip()}")


def sep(char="═", width=80):
    print(char * width)


# ── Streaming ─────────────────────────────────────────────────────────────


async def stream_turn(kwargs):
    chunks = []
    chunk_finish_reason = None
    t0 = time.time()

    response = await litellm.acompletion(**kwargs)
    async for chunk in response:
        chunks.append(chunk)
        choices = getattr(chunk, "choices", None)
        if choices and choices[0].finish_reason:
            chunk_finish_reason = choices[0].finish_reason

    elapsed = time.time() - t0
    final = litellm.stream_chunk_builder(chunks) if chunks else None
    return final, chunks, chunk_finish_reason, elapsed


def print_turn(label, final, chunks, chunk_finish_reason, elapsed):
    msg = final.choices[0].message if final else None

    print(f"\n{'─' * 60}")
    print(f"  {label}")
    print(f"{'─' * 60}")
    print(f"  chunks={len(chunks)}  elapsed={elapsed:.1f}s")
    print(
        f"  chunk_finish_reason={chunk_finish_reason}  "
        f"builder_finish_reason={final.choices[0].finish_reason if final else 'N/A'}"
    )

    # Dump the complete message object — every attribute, nothing filtered
    dump_all_attrs(msg, "MESSAGE (all attributes)")

    # Dump usage separately for clarity
    if final and final.usage:
        dump_all_attrs(final.usage, "USAGE (all attributes)")
    else:
        print("\n  USAGE: None")

    return msg


def build_assistant_dict(msg):
    """Convert litellm message to dict, preserving ALL attributes.

    No allowlist — dumps every attribute from the message object.
    This ensures we never silently drop a field litellm added.
    """
    d = {}
    for k, v in vars(msg).items():
        if k.startswith("_"):
            # Include private attrs but strip the underscore prefix
            # (e.g. _cache_read_input_tokens -> cache_read_input_tokens)
            # Skip truly internal ones like __pydantic
            if k.startswith("__"):
                continue
            # Don't include pydantic internals
            if "pydantic" in k:
                continue
        d[k] = v

    # Tool calls need conversion from objects to dicts for litellm.
    # Use model_dump() to get all fields including model_extra, then
    # recursively convert any nested pydantic models. No allowlist.
    if d.get("tool_calls"):
        d["tool_calls"] = [
            t.model_dump(exclude_none=False, exclude_unset=False)
            if hasattr(t, "model_dump")
            else t
            for t in d["tool_calls"]
        ]

    return d


def build_tool_results(msg, scatter_b64):
    """Build tool results for all tool calls. First gets image, rest get text."""
    analysis_text = (
        "Analysis complete.\n"
        "- Mean latency: 50.2ms (normal range)\n"
        "- Memory usage: linearly correlated (r=0.87)\n"
        "- ANOMALY: 5 requests at ~90ms latency show abnormally LOW memory (~50MB)\n"
        "- Likely cause: connection pooling bypass under high latency\n"
        "- See attached scatter plot for visual confirmation."
    )
    results = []
    for i, tc in enumerate(msg.tool_calls):
        if i == 0:
            results.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": [txt(analysis_text), img(scatter_b64)],
                }
            )
        else:
            results.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": analysis_text,
                }
            )
    return results


# ── Main test flow ────────────────────────────────────────────────────────


async def test_model(model_config):
    model = model_config["model"]
    reasoning_effort = model_config.get("reasoning_effort")

    sep()
    print(f"MODEL: {model}")
    print(f"  reasoning_effort: {reasoning_effort}")
    sep("─")

    bar_b64 = make_bar_chart()
    scatter_b64 = make_scatter_plot()
    heatmap_b64 = make_heatmap()

    base_kwargs = {
        "model": model,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if reasoning_effort:
        base_kwargs["reasoning_effort"] = reasoning_effort

    # ── TURN 1: User image + text → should call tool ─────────────────
    messages = [
        {
            "role": "system",
            "content": "You are a senior SRE. Be concise. Always use analyze_metrics when asked to investigate.",
        },
        {
            "role": "user",
            "content": [
                txt(
                    "I'm seeing elevated errors. Here's the monthly error chart. "
                    "Identify the anomaly and use analyze_metrics to investigate "
                    "latency-memory correlation for that month."
                ),
                img(bar_b64),
            ],
        },
    ]

    final1, c1, cfr1, e1 = await stream_turn(
        {**base_kwargs, "messages": messages, "tools": TOOLS}
    )
    msg1 = print_turn("TURN 1: user image+text → tool call", final1, c1, cfr1, e1)

    if not (msg1 and msg1.tool_calls):
        print("\n  ⚠ No tool calls — stopping here")
        sep()
        return

    # ── TURN 2: Tool results (multimodal) → text response ────────────
    assistant1 = build_assistant_dict(msg1)
    tool_results = build_tool_results(msg1, scatter_b64)

    # Print what we're sending back — so we can see if we dropped anything
    print("\n  SENDING BACK (assistant dict):")
    for k, v in assistant1.items():
        print(f"    {k}: {dump(v, 2).lstrip()}")

    messages_t2 = messages + [assistant1] + tool_results

    final2, c2, cfr2, e2 = await stream_turn({**base_kwargs, "messages": messages_t2})
    msg2 = print_turn(
        "TURN 2: tool result (text+image) → response", final2, c2, cfr2, e2
    )

    if not msg2:
        print("\n  ⚠ No response — stopping here")
        sep()
        return

    # ── TURN 3: Second user image → tests signature round-trip ────────
    assistant2 = build_assistant_dict(msg2)
    messages_t3 = messages_t2 + [
        assistant2,
        {
            "role": "user",
            "content": [
                txt(
                    "Here's an error rate heatmap by hour and day for May. "
                    "What pattern do you see? Does it correlate with the latency anomaly?"
                ),
                img(heatmap_b64),
            ],
        },
    ]

    final3, c3, cfr3, e3 = await stream_turn({**base_kwargs, "messages": messages_t3})
    msg3 = print_turn(
        "TURN 3: second user image → response (signature round-trip)",
        final3,
        c3,
        cfr3,
        e3,
    )

    if not msg3:
        print("\n  ⚠ No response — stopping here")
        sep()
        return

    # ── TURN 4: Text-only summary → multi-turn image memory ──────────
    assistant3 = build_assistant_dict(msg3)
    messages_t4 = messages_t3 + [
        assistant3,
        {
            "role": "user",
            "content": "Summarize across all three visuals (bar chart, scatter, heatmap). What's the root cause?",
        },
    ]

    final4, c4, cfr4, e4 = await stream_turn({**base_kwargs, "messages": messages_t4})
    print_turn("TURN 4: text summary → multi-turn image memory", final4, c4, cfr4, e4)

    sep()


async def main():
    filters = [f.lower() for f in sys.argv[1:]]
    models = MODELS
    if filters:
        models = [m for m in MODELS if any(f in m["model"].lower() for f in filters)]
        if not models:
            print(f"No models match filters: {filters}")
            print(f"Available: {[m['model'] for m in MODELS]}")
            return

    print(f"Testing {len(models)} model(s):\n")
    for m in models:
        print(f"  {m['model']}")
    print()

    for model_config in models:
        try:
            await test_model(model_config)
        except Exception:
            sep()
            print(f"FAILED: {model_config['model']}")
            traceback.print_exc()
            sep()
        print()


if __name__ == "__main__":
    asyncio.run(main())
