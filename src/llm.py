from __future__ import annotations

import os

from langchain_openai import ChatOpenAI

from agent.messages import build_user_message


def chat_model(model_name: str, base_url: str, max_tokens: int = 512) -> ChatOpenAI:
    """Chat model on any OpenAI-compatible endpoint (local vLLM, OpenRouter, ...).

    A, B and C all go through here, so the three settings differ only in the
    prompt and the retrieved context — not in the serving stack or the sampling.
    """
    return ChatOpenAI(
        model=model_name,
        base_url=base_url,
        api_key=os.getenv("LLM_API_KEY", "EMPTY"),
        max_tokens=max_tokens,
        temperature=0.0,
    )


class VLMClient:
    """Single-shot image + prompt → answer, for the A and B baselines.

    The agent uses ``chat_model`` directly instead: it needs the chat model
    itself to bind tools and carry a multi-turn message list.
    """

    def __init__(self, model_name: str, base_url: str, max_tokens: int = 128):
        self.llm = chat_model(model_name, base_url, max_tokens)
        print(f"VLM client ready: {model_name} @ {base_url}")

    def generate_response(self, image_path: str, prompt_text: str) -> str:
        content = self.llm.invoke([build_user_message(image_path, prompt_text)]).content
        return content if isinstance(content, str) else str(content)
