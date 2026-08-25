from __future__ import annotations

import time

from PIL import Image
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call
from langchain_core.messages import AIMessage, ToolMessage
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

    On Qwen3-VL-8B it is inert, and measurably so: the same variant run with and
    without it scored 0.318 and 0.321, called no tool in 0.7% of examples either
    way (those 0.7% being the runs that errored, not choices), and passed the
    question verbatim 69.0% against 69.7%. The model retrieves on its own. Kept
    on by default anyway — it costs nothing here and every run so far was
    measured with it, so turning it off would break comparability for no gain.
    Worth switching off for any model whose chat template handles tools, if a
    turn of reasoning before the first call is ever wanted.
    """
    @wrap_model_call
    def middleware(request, handler):
        if not any(isinstance(m, ToolMessage) for m in request.messages):
            request = request.override(tool_choice="required")
        return handler(request)

    return middleware


def require_distinct_names(min_names=2, max_calls=4, force_first=True):
    """After the first pooled search, make the agent enter the KB by a new name.

    Left to itself the agent takes one shot at naming and never takes another:
    across every variant `lookup_article` came out at 1.00-1.07 calls per
    example, so the candidate pool was whatever the first guess plus the image
    index produced. That caps the pool at 45.6% of the gold articles, and the
    pipeline reaches 46.6% with a single naming call — which is why no variant
    so far has beaten it.

    Telling the agent to try again does not work: it was told, and a `verify`
    tool reported NOT CONFIRMED on 64.5% of examples while the agent changed
    name on 0.2% of those. So the second attempt is imposed here instead, the
    same way the first tool call is. One guess resolves to the gold article
    11.6% of the time with this model; two different guesses sample more of the
    84.2% that names can reach at all.

    ``max_calls`` stops the forcing from looping when the model keeps offering
    names that do not resolve — the tool refuses repeats, but nothing stops it
    proposing new useless ones.
    """
    @wrap_model_call
    def middleware(request, handler):
        if not any(isinstance(m, ToolMessage) for m in request.messages):
            if force_first:
                request = request.override(tool_choice="required")
            return handler(request)

        names, calls, searched = set(), 0, False
        for m in request.messages:
            if not isinstance(m, AIMessage):
                continue
            for tc in m.tool_calls or []:
                if tc["name"] == "lookup_article":
                    calls += 1
                    n = (tc.get("args") or {}).get("name")
                    if isinstance(n, str) and n.strip():
                        names.add(n.strip().lower())
                elif tc["name"] == "search_paragraphs":
                    searched = True

        if searched and len(names) < min_names and calls < max_calls:
            return handler(request.override(tool_choice="lookup_article"))
        return handler(request)

    return middleware


class AgenticRAG:
    """Runs the agentic RAG loop for one example at a time.

    Tools retrieve on demand and grow a per-example working set of articles; the
    model decides whether to enter by name or by image, and which to read.
    """

    def __init__(self, llm, retriever, kb, reranker, top_n=20, top_k=20,
                 max_iterations=8, force_first=True, min_names=1):
        self.llm = llm
        self.retriever = retriever
        self.kb = kb
        self.reranker = reranker
        self.top_n = top_n
        self.top_k = top_k
        self.max_iterations = max_iterations
        self.force_first = force_first
        self.min_names = min_names

    def _middleware(self):
        """Two regimes: force only the first call, or also force a second name."""
        if self.min_names > 1:
            return [require_distinct_names(self.min_names,
                                           force_first=self.force_first)]
        return [force_first_tool()] if self.force_first else []

    def run(self, image_path: str, question: str) -> AgentRun:
        t0 = time.time()
        image = Image.open(image_path).convert("RGB")
        agent = create_agent(
            model=self.llm,
            tools=build_tools(self.retriever, self.kb, self.reranker, image,
                              top_n=self.top_n, top_k=self.top_k),
            system_prompt=SYSTEM_PROMPT,
            middleware=self._middleware(),
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
