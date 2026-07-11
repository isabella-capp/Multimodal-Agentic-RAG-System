"""Demo CLI script: visual retrieval → KB lookup → paragraph reranking.

Usage example::

    python retrival.py \
        --test_image /path/to/image.jpg \
        --test_query "What populations are typically outcrossing?"
"""

import argparse

from PIL import Image

from retriever import Retriever
from knowledge_base import KnowledgeBase

BASE_FOLDER = "/work/cvcs2026/encyclopedic"


def setup_args():
    parser = argparse.ArgumentParser(description="Pipeline Multimodale (Baseline 2)")
    parser.add_argument(
        "--img_index_path",
        type=str,
        default=f"{BASE_FOLDER}/knn.index",
        help="Path all'indice FAISS",
    )
    parser.add_argument(
        "--img_index_json_path",
        type=str,
        default=f"{BASE_FOLDER}/knn.json",
        help="Path alla lista mappata",
    )
    parser.add_argument(
        "--kb_path",
        type=str,
        default=f"{BASE_FOLDER}/encyclopedic_kb_wiki.db",
        help="Path ai testi di Wiki",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=1,
        help="Quante entità visive cercare (Stage 1)",
    )
    parser.add_argument(
        "--test_image",
        type=str,
        required=True,
        help="Path all'immagine di test da dare in pasto al sistema",
    )
    parser.add_argument(
        "--test_query",
        type=str,
        required=True,
        help="La domanda dell'utente",
    )
    return parser.parse_args()


def main():
    args = setup_args()

    # Inizializzazione
    print("--- INIZIALIZZAZIONE SISTEMA ---")
    retriever = Retriever(
        img_index_path=args.img_index_path,
        img_index_json_path=args.img_index_json_path,
        top_k=args.top_k,
    )
    kb = KnowledgeBase(args.kb_path)

    # Carichiamo l'immagine utente
    user_image = Image.open(args.test_image).convert("RGB")
    user_query = args.test_query

    print(f"\n--- INIZIO PIPELINE ---")
    print(f"Domanda Utente: '{user_query}'")

    # ==========================================
    # STAGE 1: Retrieval Visivo (Trova l'Entità)
    # ==========================================
    print("\n[Stage 1] Ricerca visiva su FAISS in corso...")
    results = retriever.retrieve(user_image, user_query)

    if not results:
        print("Nessun risultato trovato su FAISS.")
        return

    # Prendiamo il miglior risultato (Top 1)
    best_match = results[0]
    print(f"-> Entità trovata: {best_match['title']}")
    print(f"-> URL Wikipedia: {best_match['wiki_url']}")
    print(f"-> Score Visivo: {best_match['score']:.4f}")

    # ==========================================
    # STAGE 2: Estrazione e Reranking Testuale
    # ==========================================
    print("\n[Stage 2] Estrazione paragrafi dalla Knowledge Base...")
    all_paragraphs = kb.get_paragraphs_by_url(best_match["wiki_url"])
    print(f"-> Trovati {len(all_paragraphs)} paragrafi grezzi per questa pagina.")

    if not all_paragraphs:
        print("-> Nessun testo disponibile per questa entità.")
        return

    print(f"\n[Stage 3] Reranking semantico dei paragrafi in base alla domanda...")
    # Filtriamo tenendo solo i 3 paragrafi più pertinenti
    best_paragraphs = retriever.rerank_paragraphs(user_query, all_paragraphs, top_n=3)

    print("\n=== RISULTATO FINALE (Contesto per Qwen) ===")
    for i, para in enumerate(best_paragraphs, 1):
        print(f"\n[Paragrafo Selezionato {i}]:")
        # Stampiamo solo i primi 300 caratteri per non intasare il terminale
        print(f"{para[:300]}...")


if __name__ == "__main__":
    main()