# retrieval/retrieval_system.py
import os
import faiss
import pickle
from sentence_transformers import SentenceTransformer

class RetrievalSystem:
    def __init__(self, chunk_dir, index_path='faiss.index', mapping_path='id_mapping.pkl'):
        self.chunk_dir = chunk_dir
        
        # Supports multiple languages, though we have only finnish documents at the moment
        self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')  
        self.index_path = index_path
        self.mapping_path = mapping_path
        self.dimension = self.model.get_sentence_embedding_dimension()
        if os.path.exists(self.index_path) and os.path.exists(self.mapping_path):
            self.index = faiss.read_index(self.index_path)
            with open(self.mapping_path, 'rb') as f:
                self.id_mapping = pickle.load(f)
        else:
            self.index = faiss.IndexFlatL2(self.dimension)
            self.id_mapping = {}
            self.build_index()
    
    def build_index(self):
        for idx, chunk_file in enumerate(os.listdir(self.chunk_dir)):
            if chunk_file.endswith('.txt'):
                with open(os.path.join(self.chunk_dir, chunk_file), 'r', encoding='utf-8') as f:
                    text = f.read()
                embedding = self.model.encode(text)
                self.index.add(embedding.reshape(1, -1))
                self.id_mapping[idx] = text
        faiss.write_index(self.index, self.index_path)
        with open(self.mapping_path, 'wb') as f:
            pickle.dump(self.id_mapping, f)
    
    def retrieve(self, query, top_k=5):
        query_embedding = self.model.encode(query).reshape(1, -1)
        distances, indices = self.index.search(query_embedding, top_k)
        results = []
        for idx in indices[0]:
            results.append(self.id_mapping[idx])
        return results
