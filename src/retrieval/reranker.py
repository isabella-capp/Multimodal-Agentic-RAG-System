import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class CrossEncoderReranker:
    """Cross-encoder paragraph reranker.

    Scores each ``(query, paragraph)`` pair jointly with a sequence-classification
    model (e.g. ``BAAI/bge-reranker-base``) and returns the *top_n* paragraphs by
    relevance. Unlike a bi-encoder, the query and paragraph attend to each other,
    and the full paragraph (up to ``max_length`` tokens) is used.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-base",
        device: str | None = None,
        max_length: int = 512,
    ):
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.max_length = max_length

        print(f"Loading cross-encoder reranker {model_name} …")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = (
            AutoModelForSequenceClassification.from_pretrained(model_name)
            .to(self.device)
            .eval()
        )
        print(f"Cross-encoder loaded on {self.device}.")

    @torch.inference_mode()
    def rerank(
        self, query: str, paragraphs: list[str], top_n: int = 3, batch_size: int = 16
    ) -> list[str]:
        """Return the *top_n* paragraphs most relevant to the query."""
        if not paragraphs:
            return []
        if len(paragraphs) <= top_n:
            return paragraphs

        scores: list[float] = []
        for i in range(0, len(paragraphs), batch_size):
            batch = paragraphs[i : i + batch_size]
            inputs = self.tokenizer(
                [[query, p] for p in batch],
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            logits = self.model(**inputs).logits.view(-1)
            scores.extend(logits.float().tolist())

        order = sorted(range(len(paragraphs)), key=lambda i: scores[i], reverse=True)
        return [paragraphs[i] for i in order[:top_n]]
