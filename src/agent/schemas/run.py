from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, ToolMessage
from pydantic import BaseModel, ConfigDict, Field, computed_field


class AgentStep(BaseModel):
    """One tool invocation in the agent loop (a retrieval tool call)."""

    order: int  # 1-based position in the loop — the "when"
    tool: str
    arguments: dict[str, Any] = Field(default_factory=dict)  # e.g. {"query": ...}
    observation: str = ""

    @computed_field  # type: ignore[prop-decorator]
    @property
    def miss(self) -> bool:
        """The tool found nothing — every such message starts "No …"/"Unknown …"."""
        return self.observation.startswith(("No ", "Unknown "))

    @computed_field  # type: ignore[prop-decorator]
    @property
    def num_paragraphs(self) -> int:
        return self.observation.count("[Paragraph ")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def num_articles(self) -> int:
        """Article titles listed by a lookup/search step."""
        return 0 if self.miss else sum(
            1 for line in self.observation.splitlines() if line.startswith("- ")
        )


class AgentRun(BaseModel):
    """Full result + metrics of running the agent on one example."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    prediction: str | None = None
    steps: list[AgentStep] = Field(default_factory=list)
    elapsed_seconds: float = 0.0
    error: str | None = None
    # Raw LangGraph messages, kept only for tracing; never serialised.
    raw: list[Any] = Field(default_factory=list, exclude=True, repr=False)

    @classmethod
    def from_messages(cls, messages: list) -> "AgentRun":
        """Build a run from a LangGraph ``{"messages": [...]}`` result.

        Pairs each tool call with its observation (by ``tool_call_id``) into an
        ordered list of steps, and takes the last non-tool assistant message as
        the prediction.
        """
        calls: dict[str, tuple[str, dict]] = {}
        for m in messages:
            if isinstance(m, AIMessage):
                for tc in m.tool_calls or []:
                    calls[tc["id"]] = (tc["name"], tc.get("args", {}))

        steps: list[AgentStep] = []
        for m in messages:
            if isinstance(m, ToolMessage):
                name, args = calls.get(m.tool_call_id, (m.name, {}))
                steps.append(
                    AgentStep(
                        order=len(steps) + 1,
                        tool=name,
                        arguments=args,
                        observation=str(m.content),
                    )
                )

        prediction: str | None = None
        for m in reversed(messages):
            if isinstance(m, AIMessage) and not m.tool_calls:
                prediction = m.content if isinstance(m.content, str) else str(m.content)
                break

        return cls(prediction=prediction, steps=steps)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def tool_called(self) -> bool:
        return len(self.steps) > 0

    @computed_field  # type: ignore[prop-decorator]
    @property
    def num_tool_calls(self) -> int:
        return len(self.steps)

    @property
    def paragraphs_used(self) -> int:
        """Paragraphs actually surfaced to the model across all tool calls."""
        return sum(s.num_paragraphs for s in self.steps)

    @property
    def tool_sequence(self) -> list[str]:
        return [s.tool for s in self.steps]
