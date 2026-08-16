from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langgraph.graph import END, StateGraph
from PIL import Image

from agent.prompts import XML_SYSTEM_PROMPT
from agent.protocols import KnowledgeBase, Reranker, Retriever
from agent.schemas import AgentStep
from agent.tools import _format
from agent.xml_parser import parse_action


# ── graph state ──────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    """Mutable state threaded through every node of the search graph.

    ``messages`` and ``steps`` use ``operator.add`` as their reducer, so nodes
    can return just the *new* items to append rather than the whole list.
    All other fields use last-writer-wins (the default).
    """
    messages: Annotated[list[Any], operator.add]
    steps: Annotated[list[AgentStep], operator.add]
    search_count: int        # number of <search> calls completed
    parse_retries: int       # consecutive malformed outputs
    prediction: str | None   # set when <answer> is successfully parsed
    error: str | None        # set on abnormal termination
    past_queries: list[str]      
    search_results: list[str]
    
# ── graph factory ─────────────────────────────────────────────────────────────

def build_search_graph(
    retriever: Retriever,
    kb: KnowledgeBase,
    reranker: Reranker,
    llm: BaseChatModel,
    image: Image.Image,
    top_n: int = 5,
    max_iterations: int = 6,
    max_parse_retries: int = 2,
):
    """Build and compile the XML-based search graph for the search pipeline.

    Instead of LLM tool-calling (which requires JSON and ``tool_choice``
    hacks), the model outputs plain ``<search>``/``<answer>`` XML tags.
    The graph parses them and routes accordingly, feeding explicit error
    messages back to the model on any format violation.

    Parameters
    ----------
    retriever, kb, reranker:
        Retrieval stack — same objects reused across examples.
    llm:
        Chat model (no tool bindings needed).
    image:
        Query image; the visual candidate pool is built once and memoised.
    top_n:
        Paragraphs returned per ``<search>`` call.
    max_iterations:
        Maximum ``<search>`` calls allowed; hitting this limit ends the run
        with ``error="max_iterations_reached"``.
    max_parse_retries:
        Consecutive malformed-output budget (across parse errors *and*
        premature ``<answer>`` attempts) before ``error="parse_failed"``.

    Returns
    -------
    A compiled ``langgraph.graph.CompiledGraph`` ready for ``.invoke()``.
    """
    # ── memoised visual pool ─────────────────────────────────────────────────
    _pool_cache: dict = {}

    def _get_pool() -> list[str]:
        if "pool" not in _pool_cache:
            articles = retriever.retrieve(image=image, question=None)
            _pool_cache["pool"] = [
                p for a in articles
                for p in kb.get_paragraphs_by_url(wiki_url=a["wiki_url"])
            ]
        return _pool_cache["pool"]

    # ── helper ───────────────────────────────────────────────────────────────

    def _last_text(state: AgentState) -> str:
        """Extract plain text from the last message in the history."""
        last = state["messages"][-1]
        c = last.content
        return c if isinstance(c, str) else str(c)

    # ── nodes ────────────────────────────────────────────────────────────────

    def call_model(state: AgentState) -> dict:
        """Ask the LLM for its next action (search or answer)."""
        response = llm.invoke(state["messages"])
        return {"messages": [response]}

    def execute_search(state: AgentState) -> dict:
        """Run the image-grounded retrieval and return results as a feedback message.

        Implements a **sliding window** on ``search_results``: only the
        observations from the last 2 searches are kept in state, preventing
        unbounded context growth.
        """
        _, query = parse_action(_last_text(state))

        pool = _get_pool()
        if pool:
            results = reranker.rerank(query, pool, top_n=top_n)
            obs = _format(results) if results else "No relevant paragraphs found."
        else:
            obs = "No articles found for this image."

        # ── sliding window (keep only the last 2 search results) ─────────────
        window: list[str] = state["search_results"][-1:] + [obs]   # max 2 entries

        step = AgentStep(
            order=state["search_count"] + 1,
            tool="search_by_image",
            arguments={"query": query},
            observation=obs,
        )
        feedback = HumanMessage(
            content=f"[Wikipedia search results for: {query!r}]\n{obs}"
        )
        return {
            "messages": [feedback],
            "steps": [step],
            "search_count": state["search_count"] + 1,
            "parse_retries": 0,   # reset on a valid action
            "past_queries": state["past_queries"] + [query],
            "search_results": window,
        }

    def finalize_answer(state: AgentState) -> dict:
        """Extract the answer text and store it as the run's prediction."""
        _, answer = parse_action(_last_text(state))
        return {"prediction": answer}

    def handle_parse_error(state: AgentState) -> dict:
        """Inject explicit retry feedback when the model produces unparseable output."""
        last_ai_message = state["messages"][-1].content
        wrong_output = last_ai_message.strip() if isinstance(last_ai_message, str) else str(last_ai_message)
        
        # Costruisci un messaggio dinamico che suggerisce la correzione esatta
        feedback = HumanMessage(content=(
            f"FORMAT ERROR. You wrote plain text: '{wrong_output}'.\n"
            f"If this was your final answer, you MUST wrap it exactly like this: <answer>{wrong_output}</answer>.\n"
            f"Do NOT call <search> again if you already have the answer. Just output the <answer> tag."
        ))
        
        return {
            "messages": [feedback],
            "parse_retries": state["parse_retries"] + 1,
        }

    def handle_early_answer(state: AgentState) -> dict:
        """Redirect the model when it tries to answer without searching first."""
        feedback = HumanMessage(content=(
            "You used <answer> without calling <search> first.\n"
            "You MUST search Wikipedia before answering. Try:\n"
            "  <search>your search query here</search>"
        ))
        return {
            "messages": [feedback],
            "parse_retries": state["parse_retries"] + 1,
        }

    def handle_duplicate_query(state: AgentState) -> dict:
        """Inject an error when the model repeats an already-seen query.

        Instead of calling FAISS/the reranker, the graph short-circuits back
        to ``call_model`` with an explicit instruction to diversify keywords.
        This counts as a parse retry to avoid infinite loops.
        """
        _, query = parse_action(_last_text(state))
        feedback = HumanMessage(content=(
            f"Errore: hai già cercato questa esatta frase ({query!r}). "
            "Usa parole chiave diverse o usa <answer> per rispondere "
            "con le informazioni che hai."
        ))
        return {
            "messages": [feedback],
            "parse_retries": state["parse_retries"] + 1,
        }

    def set_error_max_iter(state: AgentState) -> dict:
        return {"error": "max_iterations_reached"}

    def set_error_parse(state: AgentState) -> dict:
        return {"error": "parse_failed"}

    # ── routing (conditional edges) ───────────────────────────────────────────

    def route_after_model(state: AgentState) -> str:
        """Read the last model output and decide the next node."""
        action, value = parse_action(_last_text(state))
        if action == "search":
            # Guard: max iterations.
            if state["search_count"] >= max_iterations:
                return "max_iter"
            # Guard: duplicate query.
            if value in state["past_queries"]:
                return "duplicate_query"
            return "search"
        if action == "answer":
            # First-turn answer without evidence is rejected.
            return "early_answer" if state["search_count"] == 0 else "answer"
        # Model output has no recognisable XML tag.
        return "parse_error"

    def route_after_error(state: AgentState) -> str:
        """Decide whether to give the model another chance or terminate."""
        return "end" if state["parse_retries"] >= max_parse_retries else "retry"

    # ── assemble ─────────────────────────────────────────────────────────────

    g = StateGraph(AgentState)

    g.add_node("call_model",           call_model)
    g.add_node("execute_search",       execute_search)
    g.add_node("finalize_answer",      finalize_answer)
    g.add_node("handle_parse_error",   handle_parse_error)
    g.add_node("handle_early_answer",  handle_early_answer)
    g.add_node("handle_duplicate_query", handle_duplicate_query)
    g.add_node("set_error_max_iter",   set_error_max_iter)
    g.add_node("set_error_parse",      set_error_parse)

    g.set_entry_point("call_model")

    g.add_conditional_edges(
        "call_model",
        route_after_model,
        {
            "search":           "execute_search",
            "answer":           "finalize_answer",
            "early_answer":     "handle_early_answer",
            "parse_error":      "handle_parse_error",
            "max_iter":         "set_error_max_iter",
            "duplicate_query":  "handle_duplicate_query",
        },
    )
    g.add_edge("execute_search", "call_model")

    # Error handlers share the same retry-or-end routing.
    for error_node in ("handle_parse_error", "handle_early_answer", "handle_duplicate_query"):
        g.add_conditional_edges(
            error_node,
            route_after_error,
            {"retry": "call_model", "end": "set_error_parse"},
        )

    g.add_edge("finalize_answer",    END)
    g.add_edge("set_error_max_iter", END)
    g.add_edge("set_error_parse",    END)

    return g.compile()


def make_initial_state(system_prompt: str, image_path: str, question: str) -> AgentState:
    """Build the initial graph state for one example.

    Importing here avoids a circular dependency (``messages`` ← ``rag`` ←
    ``graph``). Call this in ``rag.py`` just before ``graph.invoke()``.
    """
    from langchain_core.messages import SystemMessage
    from agent.messages import build_user_message

    return AgentState(
        messages=[
            SystemMessage(content=system_prompt),
            build_user_message(image_path=image_path, question=question),
        ],
        steps=[],
        search_count=0,
        parse_retries=0,
        prediction=None,
        error=None,
        past_queries=[],
        search_results=[],
    )
