"""LangChain-compatible wrapper around QwenVQAModel.

This thin adapter exposes the existing ``QwenVQAModel`` (which lives in
``vlm.qwen_model`` and is **not** modified) through LangChain's ``LLM``
interface so it can be used as the "brain" of a ReAct agent.

The wrapper holds a fixed ``image_path`` for the duration of a single
agent session — every ``_call`` invocation sends that same image together
with the textual prompt produced by the agent loop.
"""

from __future__ import annotations

from typing import Any, Optional

from langchain.llms.base import LLM
from pydantic import Field, ConfigDict


class QwenLangChainLLM(LLM):
    """LangChain LLM backed by a pre-loaded QwenVQAModel instance.

    Parameters
    ----------
    vlm : Any
        A ``QwenVQAModel`` instance (already loaded and on device).
    image_path : str
        Absolute path to the user image that accompanies every prompt.
    """

    # Pydantic v2 — allow the arbitrary QwenVQAModel object
    model_config = ConfigDict(arbitrary_types_allowed=True)

    vlm: Any = Field(exclude=True)
    image_path: str = ""

    # ------------------------------------------------------------------ #
    # LangChain interface                                                  #
    # ------------------------------------------------------------------ #

    @property
    def _llm_type(self) -> str:  # noqa: D401
        return "qwen-vqa"

    def _call(
        self,
        prompt: str,
        stop: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> str:
        """Forward *prompt* together with the fixed image to the VLM."""
        response = self.vlm.generate_response(self.image_path, prompt)
        # Respect any stop tokens requested by the agent framework
        if stop:
            for tok in stop:
                if tok in response:
                    response = response[: response.index(tok)]
        return response
