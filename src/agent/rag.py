from __future__ import annotations

import time

from typing import Callable, Any
from PIL import Image
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langgraph.errors import GraphRecursionError

from agent.messages import build_user_message
from agent.prompts import SYSTEM_PROMPT, TEXT_TOOL_SECTION
from prompts import ANSWER_FORMAT
from agent.run import AgentRun
from agent.tools import build_tools
from retrieval.bm25 import BM25Ranker

def force_first_tool() -> Any:
    """Forces the LLM to call a tool on its very first conversational turn."""
    @wrap_model_call
    def force_first_middleware(request: Any, handler: Callable) -> Any:
        if not any(isinstance(m, ToolMessage) for m in reversed(request.messages)):
            request = request.override(tool_choice="required")
        return handler(request)

    return force_first_middleware


def remind_original_question(original_question: str) -> Any:
    """Re-inject the question AND the answer format before the final answer.

    The format block sits at the end of the system prompt, tens of thousands of
    retrieved tokens back by the time the agent answers, and the agent ignores
    it: every C run so far averaged 7-18 words per answer against 2.5 for the
    same format in baseline B. That is not cosmetic — BEM pays for length, so B
    on the short format (0.359) and C at fifteen words (0.384) were never
    measured in the same regime. Repeating the format after each tool result is
    what puts them back in one.
    """
    @wrap_model_call
    def remind_question_middleware(request, handler):
        if request.messages and isinstance(request.messages[-1], ToolMessage):
            reminder = SystemMessage(content=(
                f"Reminder: keep your final answer strictly focused on the "
                f"original question: '{original_question}'\n\n{ANSWER_FORMAT}"
            ))
            request = request.override(messages=request.messages + [reminder])
        return handler(request)

    return remind_question_middleware

def require_tool_before_answer(tool_name: str, max_calls: int = 3) -> Any:
    """Do not let the agent answer until it has called ``tool_name`` once.

    The same lever as `require_distinct_names`, for the same reason: the agent
    is told when a tool would help and does not act on it — a `verify` tool
    reported NOT CONFIRMED on 64.5% of examples and the agent changed course on
    0.2% of those. Forcing the call separates "the tool does not help" from "the
    agent does not use it", which are very different conclusions.
    """
    @wrap_model_call
    def middleware(request, handler):
        if should_force(request.messages, tool_name):
            return handler(request.override(tool_choice=tool_name))
        return handler(request)

    return middleware


def open_text_gate(state: dict, tool_name: str, threshold: float) -> Any:
    """Force ``tool_name`` only where the pool looks like it lacks the answer.

    The agentic version of the pipeline's second round, and the same trigger: a
    measured score rather than the model's own sense of whether it has enough,
    which failed in five separate experiments. What differs from the pipeline is
    the query — the agent writes `Phyllanthus niruri` where the pipeline can
    only pass "this plant", and this channel lives on rare words.
    """
    @wrap_model_call
    def middleware(request, handler):
        score = state.get("top_score")
        if score is not None and score < threshold and should_force(
                request.messages, tool_name):
            return handler(request.override(tool_choice=tool_name))
        return handler(request)

    return middleware


def should_force(messages, tool_name: str) -> bool:
    """True while the agent has read a tool result but never called ``tool_name``.

    Split out of the middleware so it can be tested without a model: the last
    version of this shipped with an undefined name and failed on all 1000
    examples, which a two-line check would have caught.
    """
    if not any(isinstance(m, ToolMessage) for m in messages):
        return False        # first turn: force_first_tool's job, not ours
    called = sum(1 for m in messages if isinstance(m, AIMessage)
                 for tc in (m.tool_calls or []) if tc["name"] == tool_name)
    return called == 0


class AgenticRAG:
    """Runs the agentic RAG loop for one example at a time.

    Tools retrieve on demand and grow a per-example working set of articles; the
    model decides whether to enter by name or by image, and which to read.
    """

    def __init__(self, llm, retriever, kb, reranker, top_n=5, top_k=20,
                 bm25_top_m=50, max_iterations=8, force_first=True,
                 retrieval_mode: str = "bm25+reranker", rrf_k: int = 60,
                 with_text: bool = False, text_limit: int = 5,
                 force_text: bool = False, text_gate: float | None = None):
        self.llm = llm
        self.retriever = retriever
        self.kb = kb
        self.reranker = reranker
        self.bm25 = BM25Ranker()
        self.top_n = top_n
        self.top_k = top_k
        self.bm25_top_m = bm25_top_m
        self.max_iterations = max_iterations
        self.force_first = force_first
        self.retrieval_mode = retrieval_mode
        self.rrf_k = rrf_k
        self.with_text = with_text
        self.text_limit = text_limit
        self.force_text = force_text
        self.text_gate = text_gate

    def _middleware(self, question: str, state: dict) -> list[Any]:
        middlewares = []
        if self.force_first:
            middlewares.append(force_first_tool())
        if self.with_text and self.text_gate is not None:
            middlewares.append(open_text_gate(state, "search_by_text", self.text_gate))
        elif self.force_text and self.with_text:
            middlewares.append(require_tool_before_answer("search_by_text"))
        middlewares.append(remind_original_question(question))
        return middlewares

    def run(self, image_path: str, question: str) -> AgentRun:
        t0 = time.time()

        state: dict = {}   # per example: the tools write into it, the middleware reads

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            return AgentRun(error=f"Image load error: {str(e)}")
        
        agent = create_agent(
            model=self.llm,
            tools=build_tools(self.retriever, self.kb, self.reranker, self.bm25,
                              image, top_n=self.top_n, top_k=self.top_k,
                              bm25_top_m=self.bm25_top_m,
                              retrieval_mode=self.retrieval_mode,
                              rrf_k=self.rrf_k,
                              with_text=self.with_text,
                              text_limit=self.text_limit,
                              state=state),
            system_prompt=SYSTEM_PROMPT + (TEXT_TOOL_SECTION if self.with_text else ""),
            middleware=self._middleware(question, state),
        )

        try:
            out = agent.invoke(
                {"messages": [build_user_message(image_path, question)]},
                config={"recursion_limit": 2 * self.max_iterations + 1},
            )
            run = AgentRun.from_messages(out["messages"])
            run.messages = out["messages"]
        except GraphRecursionError:
            run = AgentRun(error="recursion_limit")
        except Exception as e:
            run = AgentRun(error=str(e))

        run.elapsed_seconds = round(time.time() - t0, 2)
        return run
