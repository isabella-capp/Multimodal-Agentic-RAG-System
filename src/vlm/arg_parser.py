import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="Qwen2.5-VL inference on Encyclopedic-VQA"
    )
    parser.add_argument("--output", default="outputs/predictions.jsonl")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--use-retrieval",
        action="store_true",
        help="Enable visual retrieval + KB context augmentation.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=1,
        help="Number of FAISS nearest neighbours to retrieve.",
    )
    parser.add_argument(
        "--rerank-top-n",
        type=int,
        default=3,
        help="Paragraphs to keep after reranking (or the first N with --no-rerank; <=0 keeps all).",
    )
    parser.add_argument(
        "--no-rerank",
        action="store_true",
        help="Skip paragraph reranking; use the first --rerank-top-n paragraphs directly.",
    )
    parser.add_argument(
        "--reranker",
        choices=["clip", "cross"],
        default="clip",
        help="Paragraph reranker: 'clip' (EVA-CLIP bi-encoder) or 'cross' (cross-encoder).",
    )
    parser.add_argument(
        "--debug-samples",
        type=int,
        default=3,
        help="Print a detailed pipeline trace for the first N processed examples.",
    )
    return parser.parse_args()
