import faiss
import json
base_folder = "/work/cvcs2026/encyclopedic"


if __name__ == "__main__":
    # Carica solo il JSON, l'indice FAISS lo lasciamo stare per ora
    with open(f"{base_folder}/encyclopedic_kb_wiki.json", "r") as f:
        # Usiamo un iteratore o leggiamo solo un pezzo se fosse una lista enorme
        data = json.load(f)

    print(f"Tipo di struttura: {type(data)}")
    print(f"{json.dumps(data[:2], indent=2)}")  # Stampa solo i primi 1000 caratteri per non saturare il terminale