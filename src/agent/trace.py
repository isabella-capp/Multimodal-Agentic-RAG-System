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
    """Pretty-print one full agent run: question → action(s) → answer.

    Handles both pipelines:

    * **search** (XML graph): ``AIMessage`` content contains ``<search>`` or
      ``<answer>`` tags; results arrive as plain ``HumanMessage`` with the
      ``[Wikipedia search results for: ...]`` prefix.
    * **research** (tool-calling): the classic ``AIMessage`` with
      ``tool_calls`` → ``ToolMessage`` → final ``AIMessage`` pattern.
    """
    lines = ["=" * 78]
    for m in messages:
        if isinstance(m, SystemMessage):
            lines += ["[SYSTEM]", _text(m.content).strip(), ""]

        elif isinstance(m, HumanMessage):
            text = _text(m.content).strip()
            # Distinguish the initial user question from search-result feedback.
            if text.startswith("[Wikipedia search results"):
                lines += ["[SEARCH RESULT]", text, ""]
            elif text.startswith("Your response did not") or text.startswith("You used <answer>") or text.startswith("You MUST"):
                lines += ["[PARSE ERROR FEEDBACK]", text, ""]
            else:
                lines += ["[USER]", text, ""]

        elif isinstance(m, AIMessage):
            raw = _text(m.content).strip()
            if m.tool_calls:
                # Research pipeline: tool-calling format.
                lines.append("[ASSISTANT → tool call]")
                if raw:
                    lines.append(f"  reasoning: {raw}")
                for tc in m.tool_calls:
                    lines.append(f"  call: {tc['name']}(args={tc.get('args', {})})")
            elif "<search>" in raw.lower():
                # Search pipeline: XML search action.
                lines += ["[ASSISTANT → <search>]", f"  {raw}"]
            elif "<answer>" in raw.lower():
                # Search pipeline: XML answer action.
                lines += ["[ASSISTANT → <answer>]", f"  {raw}"]
            else:
                # Research pipeline: free-text final answer.
                lines += ["[ASSISTANT → final answer]", f"  {raw}"]
            lines.append("")

        elif isinstance(m, ToolMessage):
            # Research pipeline tool results.
            lines += [f"[TOOL RESULT] ({m.name})", _text(m.content).strip(), ""]

    lines.append("=" * 78)
    return "\n".join(lines)
