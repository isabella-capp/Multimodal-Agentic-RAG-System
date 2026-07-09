import torch
import torch.nn.functional as F
import faiss
import json
import numpy as np
from transformers import CLIPImageProcessor, AutoModel, AutoTokenizer
from PIL import Image

class Retriever:
    def __init__(self, args):
        self.args = args
        self.top_k = args.top_k
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Caricamento di FAISS e Modello
        self.load_index()
        self.load_embedding_model()

    def load_index(self):
        print("Loading FAISS indices...")
        if self.args.img_index_path and self.args.img_index_json_path:
            self.img_index = faiss.read_index(self.args.img_index_path, faiss.IO_FLAG_MMAP)
            with open(self.args.img_index_json_path, "r") as f:
                self.img_values = json.load(f)
        else:
            raise ValueError("You must provide either img_index_path and img_index_json_path")
        print("FAISS indices loaded successfully.")

    def load_embedding_model(self):
        print("Loading embedding model...")
        print("Caricamento del modello EVA-CLIP...")
        self.processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.tokenizer = AutoTokenizer.from_pretrained("BAAI/EVA-CLIP-8B", trust_remote_code=True)
        self.embedding_model = AutoModel.from_pretrained(
            "BAAI/EVA-CLIP-8B", 
            torch_dtype=torch.float16, 
            trust_remote_code=True
        ).to(self.device).eval()
        print("Embedding model loaded successfully.")

    def encode_image_query(self, image: Image.Image):
        """Genera SOLO l'embedding dell'immagine (Approccio ReAG - Stage 1)."""
        image_tensor = self.processor(image, return_tensors="pt").pixel_values.to(self.device, dtype=torch.float16)
        
        with torch.no_grad():
            image_features = self.embedding_model.encode_image(image_tensor)
            
        # Normalizzazione (fondamentale per FAISS)
        image_features = F.normalize(image_features, dim=-1)
        
        return image_features.cpu().numpy().astype(np.float32)
    
    def retrieve_top_k(self, image: Image.Image):
        """Interroga FAISS usando solo l'immagine e restituisce i metadati."""
        # Genera il vettore visivo
        query_embeds = self.encode_image_query(image)
        
        # Ricerca su FAISS
        distances, indices = self.img_index.search(query_embeds, k=self.top_k)
        
        ids = indices[0]
        raw_scores = distances[0].tolist()
        
        # Recupero dei risultati dal file JSON mappato (URL, Title, Image_path)
        results = []
        for idx in ids:
            if idx != -1 and idx < len(self.img_values):
                data = self.img_values[idx]
                results.append({
                    "wiki_url": data[0],
                    "title": data[1],
                    "image_path": data[2]
                })
        
        return results, raw_scores
