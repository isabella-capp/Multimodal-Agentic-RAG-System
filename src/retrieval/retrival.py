import argparse
from PIL import Image

# Assicurati che questi import puntino ai file corretti che hai creato
from retriever import Retriever

base_folder = "/work/cvcs2026/encyclopedic"

def setup_args():
    parser = argparse.ArgumentParser(description="Pipeline Multimodale (Baseline 2)")
    parser.add_argument("--img_index_path", type=str, default=f"{base_folder}/knn.index", help="Path all'indice FAISS")
    parser.add_argument("--img_index_json_path", type=str, default=f"{base_folder}/knn.json", help="Path alla lista mappata")
    parser.add_argument("--kb_path", type=str, default=f"{base_folder}/encyclopedic_kb_wiki.json", help="Path ai testi di Wiki")
    parser.add_argument("--top_k", type=int, default=1, help="Quante entità visive cercare (Stage 1)")
    parser.add_argument("--test_image", type=str, required=True, help="Path all'immagine di test da dare in pasto al sistema")
    parser.add_argument("--test_query", type=str, required=True, help="La domanda dell'utente")
    
    return parser.parse_args()

def main():
    args = setup_args()

    # 2. Inizializziamo i moduli (I nostri "Strumenti")
    print("--- INIZIALIZZAZIONE SISTEMA ---")
    retriever = Retriever(args)
    kb = KnowledgeBase(args.kb_path)  # Assicurati di avere una classe KnowledgeBase che gestisca il caricamento dei testi
    
    # Carichiamo l'immagine utente
    user_image = Image.open(args.test_image).convert("RGB")
    user_query = args.test_query

    print(f"\n--- INIZIO PIPELINE ---")
    print(f"Domanda Utente: '{user_query}'")

    # ==========================================
    # STAGE 1: Retrieval Visivo (Trova l'Entità)
    # ==========================================
    print("\n[Stage 1] Ricerca visiva su FAISS in corso...")
    results, scores = retriever.retrieve_top_k(user_image)
    
    if not results:
        print("Nessun risultato trovato su FAISS.")
        return

    # Prendiamo il miglior risultato (Top 1)
    best_match = results[0]
    best_url = best_match['wiki_url']
    print(f"-> Entità trovata: {best_match['title']}")
    print(f"-> URL Wikipedia: {best_url}")
    print(f"-> Score Visivo: {scores[0]:.4f}")

    # ==========================================
    # STAGE 2: Estrazione e Reranking Testuale
    # ==========================================
    print("\n[Stage 2] Estrazione paragrafi dalla Knowledge Base...")
    all_paragraphs = kb.get_paragraphs_by_url(best_url)
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
        print(f"{para[:300]}...") # Stampiamo solo i primi 300 caratteri per non intasare il terminale

if __name__ == "__main__":
    main()