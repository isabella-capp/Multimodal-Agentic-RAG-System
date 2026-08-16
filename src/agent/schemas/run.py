from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, ToolMessage
from pydantic import BaseModel, ConfigDict, Field, computed_field


class AgentStep(BaseModel):
    """One tool invocation in the agent loop (a retrieval tool call)."""

    order: int  # 1-based position in the loop — the "when"
    tool: str
    arguments: dict[str, Any] = Field(default_factory=dict)  # e.g. {\"query\": ...}
    observation: str = ""


class AgentRun(BaseModel):
    """Full result + metrics of running the agent on one example."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    prediction: str | None = None
    steps: list[AgentStep] = Field(default_factory=list)
    elapsed_seconds: float = 0.0
    error: str | None = None
    # Raw messages / state, kept only for tracing; never serialised.
    raw: list[Any] = Field(default_factory=list, exclude=True, repr=False)

    @classmethod
    def from_messages(cls, messages: list) -> "AgentRun":
        """Build a run from a LangGraph ``{\"messages\": [...]}`` result.

        Used exclusively by the **research** pipeline, which still runs on
        LangChain's tool-calling agent.  Pairs each tool call with its
        observation and takes the last free-text AI message as the prediction.
        """
        calls: dict[str, tuple[str, dict]] = {}
        for m in messages:
            if isinstance(m, AIMessage):
                for tc in m.tool_calls or []:
                    tool_call_id = tc.get("id")
                    if tool_call_id is None:
                        continue
                    calls[tool_call_id] = (tc["name"], tc.get("args", {}))

        steps: list[AgentStep] = []
        for m in messages:
            if isinstance(m, ToolMessage):
                name, args = calls.get(m.tool_call_id, (m.name, {}))

                if name is None:
                    continue
                
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

    @classmethod
    def from_graph_state(cls, state: dict) -> "AgentRun":
        """Build a run from the final state dict produced by the XML search graph.

        The graph stores ``prediction``, ``steps``, and ``error`` directly in
        its state, so no message parsing is needed.
        """
        return cls(
            prediction=state.get("prediction"),
            steps=state.get("steps", []),
            error=state.get("error"),
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
