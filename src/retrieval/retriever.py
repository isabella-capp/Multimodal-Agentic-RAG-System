import torch
import torch.nn.functional as F
import faiss
import json
import numpy as np
from transformers import CLIPImageProcessor, AutoModel, AutoTokenizer
from PIL import Image


class Retriever:
    """Visual retriever based on EVA-CLIP + FAISS.

    Encodes a user image into an embedding, searches a FAISS index for the
    top-k most similar images, and returns the associated Wikipedia metadata.
    Optionally reranks retrieved paragraphs by textual similarity to a query.
    """

    def __init__(
        self,
        img_index_path: str,
        img_index_json_path: str,
        top_k: int = 1,
        device: str | None = None,
    ):
        """
        Parameters
        ----------
        img_index_path : str
            Path to the FAISS index file (e.g. ``knn.index``).
        img_index_json_path : str
            Path to the JSON file that maps FAISS indices to
            ``[wiki_url, title, image_path]`` triples.
        top_k : int
            Number of nearest neighbours to retrieve.
        device : str | None
            Torch device string. Defaults to CUDA if available.
        """
        self.top_k = top_k
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self._img_index_path = img_index_path
        self._img_index_json_path = img_index_json_path

        # Lazy-loaded resources
        self.img_index = None
        self.img_values = None
        self.processor = None
        self.tokenizer = None
        self.embedding_model = None

    # ------------------------------------------------------------------
    # Lazy loading helpers
    # ------------------------------------------------------------------

    def _ensure_index(self):
        """Load the FAISS index and its JSON mapping (once)."""
        if self.img_index is not None:
            return

        print("Loading FAISS index …")
        self.img_index = faiss.read_index(self._img_index_path, faiss.IO_FLAG_MMAP)
        with open(self._img_index_json_path, "r") as f:
            self.img_values = json.load(f)
        print(f"FAISS index loaded ({self.img_index.ntotal} vectors).")

    def _ensure_model(self):
        """Load the EVA-CLIP model, processor, and tokenizer (once)."""
        if self.embedding_model is not None:
            return

        print("Loading EVA-CLIP embedding model …")
        self.processor = CLIPImageProcessor.from_pretrained(
            "openai/clip-vit-large-patch14"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            "BAAI/EVA-CLIP-8B", trust_remote_code=True
        )
        self.embedding_model = (
            AutoModel.from_pretrained(
                "BAAI/EVA-CLIP-8B",
                torch_dtype=torch.float16,
                trust_remote_code=True,
            )
            .to(self.device)
            .eval()
        )
        print("EVA-CLIP model loaded.")

    # ------------------------------------------------------------------
    # Core methods
    # ------------------------------------------------------------------

    def encode_image(self, image: Image.Image) -> np.ndarray:
        """Encode an image into a normalised embedding vector.

        Returns a ``(1, D)`` float32 numpy array.
        """
        self._ensure_model()

        image_tensor = self.processor(image, return_tensors="pt").pixel_values.to(
            self.device, dtype=torch.float16
        )

        with torch.no_grad():
            image_features = self.embedding_model.encode_image(image_tensor)

        image_features = F.normalize(image_features, dim=-1)
        return image_features.cpu().numpy().astype(np.float32)

    def retrieve_top_k(self, image: Image.Image):
        """Search FAISS using an image and return metadata + scores.

        Returns
        -------
        results : list[dict]
            Each dict has keys ``wiki_url``, ``title``, ``image_path``.
        scores : list[float]
            Raw FAISS distances (higher = more similar for inner-product).
        """
        self._ensure_index()

        query_embeds = self.encode_image(image)
        distances, indices = self.img_index.search(query_embeds, k=self.top_k)

        ids = indices[0]
        raw_scores = distances[0].tolist()

        results = []
        for idx in ids:
            if idx != -1 and idx < len(self.img_values):
                data = self.img_values[idx]
                results.append(
                    {
                        "wiki_url": data[0],
                        "title": data[1],
                        "image_path": data[2],
                    }
                )

        return results, raw_scores

    def retrieve(self, image: Image.Image, question: str | None = None):
        """High-level retrieval: image → FAISS results with scores.

        Parameters
        ----------
        image : PIL.Image.Image
            The user query image.
        question : str | None
            The user question (currently unused in FAISS search but kept
            for API consistency and future multimodal retrieval).

        Returns
        -------
        results : list[dict]
            Each dict has ``wiki_url``, ``title``, ``image_path``, ``score``.
        """
        raw_results, scores = self.retrieve_top_k(image)

        results = []
        for res, score in zip(raw_results, scores):
            results.append({**res, "score": score})

        return results

    # ------------------------------------------------------------------
    # Paragraph reranking
    # ------------------------------------------------------------------

    def rerank_paragraphs(
        self, query: str, paragraphs: list[str], top_n: int = 3
    ) -> list[str]:
        """Rerank paragraphs by cosine similarity to the query.

        Uses the EVA-CLIP text encoder to embed both the query and each
        paragraph, then returns the *top_n* paragraphs sorted by
        descending similarity.

        Parameters
        ----------
        query : str
            The user question.
        paragraphs : list[str]
            Candidate paragraphs from the knowledge base.
        top_n : int
            Number of top paragraphs to return.

        Returns
        -------
        list[str]
            The *top_n* most relevant paragraphs.
        """
        if not paragraphs:
            return []
        if len(paragraphs) <= top_n:
            return paragraphs

        self._ensure_model()

        # Encode query
        query_emb = self._encode_text(query)

        # Encode all paragraphs
        para_embs = self._encode_text(paragraphs)

        # Cosine similarity (embeddings are already normalised)
        similarities = (para_embs @ query_emb.T).squeeze(-1)  # (N,)

        # Top-n indices
        top_indices = similarities.argsort(descending=True)[:top_n].tolist()
        return [paragraphs[i] for i in top_indices]

    def _encode_text(self, texts: str | list[str]) -> torch.Tensor:
        """Encode one or more text strings into normalised embeddings.

        Returns a ``(N, D)`` float16 tensor on ``self.device``.
        """
        if isinstance(texts, str):
            texts = [texts]

        # Tokenize with truncation to model max length
        tokens = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            text_features = self.embedding_model.encode_text(tokens["input_ids"])

        text_features = F.normalize(text_features, dim=-1)
        return text_features
