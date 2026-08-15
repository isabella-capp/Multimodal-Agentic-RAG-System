from __future__ import annotations

import os

from langchain_openai import ChatOpenAI

from agent.messages import build_user_message


class VLMClient:
    """Chat client for a model on any OpenAI-compatible endpoint (vLLM, ...).

    A, B and C all answer through this, so the three settings differ only in the
    prompt and the retrieved context — not in the serving stack. Running the
    baselines on HF ``generate`` while the agent ran on vLLM left an
    uncontrolled difference in a comparison meant to isolate the method.

    Keeps the ``generate_response`` signature of the HF model it replaces.
    """

    def __init__(self, model_name: str, base_url: str = "http://localhost:8000/v1",
                 max_tokens: int = 128):
        self.model_name = model_name
        self.llm = ChatOpenAI(
            model=model_name,
            base_url=base_url,
            api_key=os.getenv("LLM_API_KEY", "EMPTY"),
            max_tokens=max_tokens,
            temperature=0.0,
        )
        print(f"VLM client ready: {model_name} @ {base_url}")

    def generate_response(self, image_path_or_url: str, prompt_text: str) -> str:
        message = build_user_message(image_path=image_path_or_url, question=prompt_text)
        content = self.llm.invoke([message]).content
        return content if isinstance(content, str) else str(content)
