"""
Default convert_to_llm implementation.

This is the only place in liteagent with provider-specific logic.
Everything else goes through litellm provider-agnostically.

If litellm is ever replaced, this module is the boundary to update.
"""

# Fields added by liteagent that are NOT part of the LLM message format.
# The default converter strips these; everything else passes through.
_LITEAGENT_FIELDS = {
    "timestamp",
    "usage",
    "stop_reason",
    "error_message",
    "details",
    "is_error",
}


def _is_openai_model(model):
    return model.startswith("gpt") or model.startswith("openai/")


def make_default_convert(model):
    """Build the default convert_to_llm for a model.

    Strips liteagent metadata, passes everything else through — including
    thinking_blocks, reasoning_content, provider_specific_fields, and
    multimodal content blocks (text + image_url).

    For OpenAI Chat Completions models: hoists images from tool results
    into synthetic user messages, since OpenAI ignores image blocks in
    tool result content.
    """
    hoist_tool_images = _is_openai_model(model)

    def convert(messages):
        result = []
        for m in messages:
            role = m.get("role")
            if role not in ("assistant", "user", "tool"):
                continue
            msg = {k: v for k, v in m.items() if k not in _LITEAGENT_FIELDS}

            if (
                hoist_tool_images
                and role == "tool"
                and isinstance(msg.get("content"), list)
            ):
                blocks = msg["content"]
                images = [b for b in blocks if b.get("type") == "image_url"]
                if images:
                    text_parts = [b["text"] for b in blocks if b.get("type") == "text"]
                    msg["content"] = "\n".join(text_parts) or "(see attached image)"
                    result.append(msg)
                    result.append(
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "Image from tool result:"},
                                *images,
                            ],
                        }
                    )
                    continue

            result.append(msg)
        return result

    return convert
