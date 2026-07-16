from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, ToolMessage
from pydantic import BaseModel, Field, computed_field

from agent.schemas.retrieval import ArticleHit


class AgentStep(BaseModel):
    """One tool invocation in the agent loop (a `search_paragraphs` call)."""

    order: int  # 1-based position in the loop — the "when"
    tool: str
    arguments: dict[str, Any] = Field(default_factory=dict)  # e.g. {"query": ...}
    observation: str = ""


class AgentRun(BaseModel):
    """Full result + metrics of running the agent on one example."""

    prediction: str | None = None
    steps: list[AgentStep] = Field(default_factory=list)
    articles: list[ArticleHit] = Field(default_factory=list)
    num_paragraphs_pool: int = 0
    elapsed_seconds: float = 0.0
    error: str | None = None

    @classmethod
    def from_messages(
        cls,
        messages: list,
        *,
        articles: list[ArticleHit],
        num_paragraphs_pool: int,
    ) -> "AgentRun":
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

        return cls(
            prediction=prediction,
            steps=steps,
            articles=articles,
            num_paragraphs_pool=num_paragraphs_pool,
        )

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
        return sum(s.observation.count("[Paragraph ") for s in self.steps)
