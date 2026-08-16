from __future__ import annotations

import re

_SEARCH_RE = re.compile(r"<search>(.*?)</search>", re.DOTALL | re.IGNORECASE)
_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)


def parse_action(text: str) -> tuple[str, str]:
    """Parse the model's free-text XML output and return ``(action, value)``.

    The parser is deliberately lenient: it accepts leading/trailing text
    around the tag (models often "think out loud" before the tag), and is
    case-insensitive.  It always returns the *first* matching tag so that
    parallel tags do not cause surprises.

    Returns
    -------
    ("search", query)
        The model issued a ``<search>`` call with the given query.
    ("answer", text)
        The model issued an ``<answer>`` with the given text.
    ("error", reason)
        Neither tag was found; ``reason`` contains a diagnostic excerpt.
    """
    m = _SEARCH_RE.search(text)
    if m:
        return "search", m.group(1).strip()

    m = _ANSWER_RE.search(text)
    if m:
        return "answer", m.group(1).strip()

    excerpt = repr(text[:150])
    return "error", f"No <search> or <answer> tag found in: {excerpt}"
