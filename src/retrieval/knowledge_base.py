import json

class KnowledgeBase:
    def __init__(self, kb_path: str):
        """
        Inizializza la KB caricando il file JSON enciclopedico.
        kb_path: es. 'encyclopedic_kb_wiki.json'
        """
        self.kb_path = kb_path
        self.kb_data = None
        self.load_kb()

    def load_kb(self):
        print(f"Charging Knowledge Base from {self.kb_path}...")
        with open(self.kb_path, "r") as f:
            self.kb_data = json.load(f)
        print(f"Knowledge base retrieved: {len(self.kb_data)}")

    def get_paragraphs_by_url(self, wiki_url: str):
        """
        Dato un URL di Wikipedia, restituisce la lista dei paragrafi.
        Filtra le stringhe vuote.
        """
        if not self.kb_data:
            raise ValueError("Knowledge Base non caricata.")

        # Cerchiamo la entry usando l'URL come chiave
        entry = self.kb_data.get(wiki_url)
        
        if entry and "section_texts" in entry:
            # Estraiamo i testi ed eliminiamo eventuali paragrafi vuoti ""
            paragraphs = [p for p in entry["section_texts"] if p.strip()]
            return paragraphs
        
        return []