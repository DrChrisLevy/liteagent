import time
from dataclasses import dataclass, field
from typing import Callable, Literal


def _now_ms():
    return int(time.time() * 1000)


StopReason = Literal["stop", "tool_calls", "length", "error", "aborted"]


@dataclass
class Tool:
    """A tool the LLM can call. Combines definition with execution."""

    name: str  # LLM-facing name ("read_file", "bash")
    description: str  # LLM-facing description
    parameters: dict  # JSON Schema — sent to litellm as tool definition
    label: str = ""  # Human-readable UI label ("Read File")
    params_model: type = None  # Optional Pydantic BaseModel for validation + coercion
    execute: Callable = (
        None  # async def(tool_call_id, params, signal, on_update) -> ToolResult
    )


@dataclass
class ToolResult:
    """What a tool returns after execution."""

    content: list  # [{"type": "text", "text": "..."}, {"type": "image_url", ...}]
    details: dict = None  # UI-only extras (not sent to LLM)


@dataclass
class AgentContext:
    """Data snapshot the loop operates on. Created once per prompt()/continue_run()."""

    system_prompt: str  # loop reads, never mutates
    messages: list  # loop APPENDS new messages here
    tools: list[Tool] | None = None  # loop reads, never mutates


@dataclass
class AgentConfig:
    """Configuration and hooks for the loop."""

    model: str  # litellm model string

    # REQUIRED: filter/transform messages for LLM.
    # Must pass thinking_blocks/reasoning_content through on assistant messages.
    # Can be sync or async — loop awaits either.
    convert_to_llm: Callable

    # Optional hooks (all can be sync or async)
    transform_context: Callable = None  # compaction, pruning, injection
    get_steering_messages: Callable = None  # check for user interruption
    get_follow_up_messages: Callable = None  # check for queued messages

    # LLM parameters
    reasoning_effort: str = None  # "minimal"/"low"/"medium"/"high"/"xhigh" or None
    max_tokens: int = None
    temperature: float = None

    # Retry behavior — passed to litellm
    num_retries: int = None


@dataclass
class AgentState:
    """Read-only snapshot of the agent's current state."""

    system_prompt: str
    model: str
    thinking_level: str  # "off", "minimal", "low", "medium", "high", "xhigh"
    tools: list[Tool] = field(default_factory=list)
    messages: list = field(default_factory=list)
    is_streaming: bool = False
    stream_message: dict | None = None
    pending_tool_calls: set[str] = field(default_factory=set)
    error: str | None = None
