import os

BASE_FOLDER = "/work/cvcs2026/encyclopedic"

JSON_PATH = f"{BASE_FOLDER}/encyclopedic_test_subset.json"
KB_PATH = f"{BASE_FOLDER}/encyclopedic_kb_wiki.db"
IMG_INDEX_PATH = f"{BASE_FOLDER}/knn.index"
IMG_INDEX_JSON_PATH = f"{BASE_FOLDER}/knn.json"

# Offline, bge-reranker-base puts the answer paragraph in the top-20 94.0% of
# the time against 97.0% for bge-reranker-v2-m3 (top-5: 74.7% against 89.3%).
# End to end that is worth +1.4 to +1.8 points to every pipeline we tried, and
# nothing at all to the agent (0.384 -> 0.382). The default stays `base` because
# every number recorded so far was measured with it; override to compare.
# Document frequencies for the full-text channel; see load_df_cache.
TERM_DF_PATH = "outputs/retrieval/term_df.json"

CROSS_ENCODER_MODEL = os.getenv("CROSS_ENCODER_MODEL", "BAAI/bge-reranker-base")
GROUNDING_MODEL = os.getenv("GROUNDING_MODEL", "IDEA-Research/grounding-dino-tiny")
GROUNDING_DEVICE = os.getenv("GROUNDING_DEVICE", "cpu")
RETRIEVER_DEVICE = "cuda"
EF_SEARCH = 256  # HNSW search depth; the ~16 default under-retrieves
