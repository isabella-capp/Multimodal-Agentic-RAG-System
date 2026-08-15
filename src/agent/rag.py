from __future__ import annotations

import time

from PIL import Image
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call
from langchain_core.messages import ToolMessage
from langgraph.errors import GraphRecursionError

from agent.messages import build_user_message
from agent.prompts import SYSTEM_PROMPT
from agent.run import AgentRun
from agent.tools import build_tools


def force_first_tool():
    """Require a tool call on the first model turn, then leave the model free.

    Left to itself Qwen2.5-VL-3B retrieves in only 28.6% of examples and answers
    the rest from memory, which the prompt forbids. Forcing the first call is
    what keeps answers grounded, and comparable across model sizes.
    """
    @wrap_model_call
    def middleware(request, handler):
        if not any(isinstance(m, ToolMessage) for m in request.messages):
            request = request.override(tool_choice="required")
        return handler(request)

    return middleware


class AgenticRAG:
    """Runs the agentic RAG loop for one example at a time.

    Tools retrieve on demand and grow a per-example working set of articles; the
    model decides whether to enter by name or by image, and which to read.
    """

    def __init__(self, llm, retriever, kb, reranker, top_n=20, top_k=20,
                 max_iterations=8, force_first=True):
        self.llm = llm
        self.retriever = retriever
        self.kb = kb
        self.reranker = reranker
        self.top_n = top_n
        self.top_k = top_k
        self.max_iterations = max_iterations
        self.force_first = force_first

    def run(self, image_path: str, question: str) -> AgentRun:
        t0 = time.time()
        image = Image.open(image_path).convert("RGB")
        agent = create_agent(
            model=self.llm,
            tools=build_tools(self.retriever, self.kb, self.reranker, image,
                              top_n=self.top_n, top_k=self.top_k),
            system_prompt=SYSTEM_PROMPT,
            middleware=[force_first_tool()] if self.force_first else [],
        )

        try:
            out = agent.invoke(
                {"messages": [build_user_message(image_path, question)]},
                # one iteration ≈ an agent step + a tool step, +1 for the answer
                config={"recursion_limit": 2 * self.max_iterations + 1},
            )
            run = AgentRun.from_messages(out["messages"])
            run.messages = out["messages"]
        except GraphRecursionError:
            run = AgentRun(error="recursion_limit")
        except Exception as e:  # keep the batch going
            run = AgentRun(error=str(e))

        run.elapsed_seconds = round(time.time() - t0, 2)
        return run
