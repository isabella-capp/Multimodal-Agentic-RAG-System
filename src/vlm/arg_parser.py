import argparse

MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Qwen2.5-VL inference on Encyclopedic-VQA"
    )
    parser.add_argument("--output", default="outputs/predictions.jsonl")
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--legacy-prompt", action="store_true",
                        help="Prompts without the answer-format block, as before it existed.")
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
        "--use-naming",
        action="store_true",
        help="Also enter the KB by name: ask the model what the image shows and "
             "add the articles that name resolves to. Requires --use-retrieval.",
    )
    parser.add_argument(
        "--naming-limit",
        type=int,
        default=3,
        help="Articles to keep from the name lookup.",
    )
    parser.add_argument(
        "--debug-samples",
        type=int,
        default=3,
        help="Print a detailed pipeline trace for the first N processed examples.",
    )
    return parser.parse_args()
