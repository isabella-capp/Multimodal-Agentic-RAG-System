from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage


def _text(content) -> str:
    """Flatten a message content (str or list of blocks) to readable text."""
    if isinstance(content, str):
        return content
    parts = []
    for block in content or []:
        if isinstance(block, dict):
            if block.get("type") == "text":
                parts.append(block["text"])
            elif "image_url" in block or block.get("type") == "image_url":
                parts.append("<image>")
        else:
            parts.append(str(block))
    return " ".join(parts)


def format_trace(messages: list) -> str:
    """Pretty-print one full agent iteration: question → tool call(s) → tool
    results → final answer. Shows exactly what the model called, what came back,
    and the answer it synthesised."""
    lines = ["=" * 78]
    for m in messages:
        if isinstance(m, SystemMessage):
            lines += ["[SYSTEM]", _text(m.content).strip(), ""]
        elif isinstance(m, HumanMessage):
            lines += ["[USER]", _text(m.content).strip(), ""]
        elif isinstance(m, AIMessage):
            reasoning = _text(m.content).strip()
            if m.tool_calls:
                lines.append("[ASSISTANT → tool call]")
                if reasoning:
                    lines.append(f"  reasoning: {reasoning}")
                for tc in m.tool_calls:
                    lines.append(f"  call: {tc['name']}(args={tc.get('args', {})})")
            else:
                lines += ["[ASSISTANT → final answer]", f"  {reasoning}"]
            lines.append("")
        elif isinstance(m, ToolMessage):
            lines += [f"[TOOL RESULT] ({m.name})", _text(m.content).strip(), ""]
    lines.append("=" * 78)
    return "\n".join(lines)
