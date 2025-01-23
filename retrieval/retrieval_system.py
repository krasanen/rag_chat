# retrieval/retrieval_system.py
import os
import json
import time
import logging
import pickle
import numpy as np
import faiss
from openai import OpenAI

import torch
import torch.nn.functional as F
from typing import List
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RetrievalSystem:

    def __init__(
        self,
        chunk_dir: str,
        index_path: str = "faiss.index",
        mapping_path: str = "id_mapping.pkl",
        cache_file: str = "embedding_cache.json",
    ):
        """
        Initializes the RetrievalSystem with enhanced caching and embedding strategies.
        """
        # Basic initialization
        self.chunk_dir = chunk_dir
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set.")
        
        # Initialize OpenAI client
        self.openai_client = OpenAI(api_key=self.openai_api_key)
        
        self.dimension = 1536  # OpenAI's text-embedding-ada-002 has 1536 dimensions

        # Caching configuration
        self.cache_file = cache_file
        self.max_cache_size = 10000
        self.embedding_cache = self._load_cache()

        # Local embedding model as fallback
        self.local_embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

        # Initialize FAISS index
        if os.path.exists(index_path) and os.path.exists(mapping_path):
            try:
                logger.info(f"Loading FAISS index from {index_path}...")
                self.index = faiss.read_index(index_path)
                with open(mapping_path, "rb") as f:
                    self.id_mapping = pickle.load(f)
                logger.info("FAISS index and ID mapping loaded successfully.")
            except Exception as e:
                logger.error(f"Error loading FAISS index or ID mapping: {e}")
                logger.info("Rebuilding the FAISS index...")
                self.index = faiss.IndexFlatL2(self.dimension)
                self.id_mapping = {}
                self.build_index(index_path, mapping_path)
        else:
            logger.info("FAISS index or ID mapping not found. Building the index...")
            self.index = faiss.IndexFlatL2(self.dimension)
            self.id_mapping = {}
            self.build_index(index_path, mapping_path)

    def _load_cache(self) -> dict:
        """Load embedding cache from file, creating if not exists."""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return {}
        except (json.JSONDecodeError, IOError):
            return {}

    def _save_cache(self):
        """Save embedding cache, maintaining LRU-like behavior."""
        try:
            # Trim cache if it gets too large
            if len(self.embedding_cache) > self.max_cache_size:
                # Remove oldest entries
                sorted_cache = sorted(
                    self.embedding_cache.items(), 
                    key=lambda x: x[1].get('timestamp', 0)
                )
                for key, _ in sorted_cache[:len(self.embedding_cache) - self.max_cache_size]:
                    del self.embedding_cache[key]

            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.embedding_cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logging.error(f"Cache save error: {e}")

    def get_openai_embedding(
        self, 
        text: str, 
        max_retries: int = 3, 
        use_local_fallback: bool = True
    ) -> List[float]:
        """
        Enhanced embedding method with multiple strategies
        """
        # Normalize text
        text = text.strip().replace('\n', ' ')[:2000]  # Limit length

        # Check cache first
        if text in self.embedding_cache:
            return self.embedding_cache[text]['embedding']

        # Try OpenAI embedding
        try:
            response = self.openai_client.embeddings.create(input=text, 
            model="text-embedding-ada-002")
            embedding = response.data[0].embedding

            # Cache the embedding
            self.embedding_cache[text] = {
                'embedding': embedding,
                'timestamp': time.time()
            }
            self._save_cache()

            return embedding

        except Exception as openai_error:
            logging.warning(f"OpenAI embedding failed: {openai_error}")

            # Local embedding fallback
            if use_local_fallback:
                try:
                    local_embedding = self.local_embedding_model.encode(text).tolist()

                    # Cache local embedding
                    self.embedding_cache[text] = {
                        'embedding': local_embedding,
                        'timestamp': time.time()
                    }
                    self._save_cache()

                    return local_embedding
                except Exception as local_error:
                    logging.error(f"Local embedding failed: {local_error}")

        return []

    def batch_get_embeddings(
        self, 
        texts: List[str], 
        batch_size: int = 100
    ) -> List[List[float]]:
        """
        Batch embedding generation with caching and fallback
        """
        embeddings = []

        # Check cache first
        uncached_texts = []
        for text in texts:
            text = text.strip().replace('\n', ' ')[:2000]
            if text in self.embedding_cache:
                embeddings.append(self.embedding_cache[text]['embedding'])
            else:
                uncached_texts.append(text)

        # Process uncached texts in batches
        for i in range(0, len(uncached_texts), batch_size):
            batch = uncached_texts[i:i+batch_size]

            try:
                # Try OpenAI batch embedding
                response = self.openai_client.embeddings.create(input=batch, 
                model="text-embedding-ada-002")
                batch_embeddings = [item['embedding'] for item in response.data]

                # Cache batch embeddings
                for text, embedding in zip(batch, batch_embeddings):
                    self.embedding_cache[text] = {
                        'embedding': embedding,
                        'timestamp': time.time()
                    }

                embeddings.extend(batch_embeddings)

            except Exception as openai_error:
                logging.warning(f"OpenAI batch embedding failed: {openai_error}")

                # Local embedding fallback
                try:
                    local_batch_embeddings = self.local_embedding_model.encode(batch).tolist()

                    # Cache local embeddings
                    for text, embedding in zip(batch, local_batch_embeddings):
                        self.embedding_cache[text] = {
                            'embedding': embedding,
                            'timestamp': time.time()
                        }

                    embeddings.extend(local_batch_embeddings)

                except Exception as local_error:
                    logging.error(f"Local batch embedding failed: {local_error}")

        # Save cache periodically
        self._save_cache()

        return embeddings

    def build_index(self, index_path: str, mapping_path: str, batch_size: int = 10):
        """
        Builds the FAISS index from text chunks using batch embeddings, with optimized chunking strategies.
        """
        # Load cache if exists
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "r", encoding="utf-8") as cf:
                embedding_cache = json.load(cf)
        else:
            embedding_cache = {}

        texts = []
        file_indices = []
        for idx, chunk_file in enumerate(os.listdir(self.chunk_dir)):
            if chunk_file.endswith('.txt'):
                chunk_path = os.path.join(self.chunk_dir, chunk_file)
                try:
                    with open(chunk_path, "r", encoding="utf-8") as f:
                        full_text = f.read().strip()

                    # Dynamic Chunking
                    chunks = self.create_semantic_chunks(full_text)

                    for chunk in chunks:
                        if not chunk.strip():  # Skip empty chunks
                            continue
                        if chunk in embedding_cache:
                            embedding = embedding_cache[chunk]
                            if embedding:
                                embedding_vector = np.array(embedding).astype("float32")
                                self.index.add(embedding_vector.reshape(1, -1))
                                self.id_mapping[idx] = chunk
                        else:
                            texts.append(chunk)
                            file_indices.append(idx)
                            if len(texts) == batch_size:
                                embeddings = self.batch_get_embeddings(texts)
                                for i, embedding in enumerate(embeddings):
                                    if not embedding:
                                        continue
                                    embedding_cache[texts[i]] = embedding
                                    embedding_vector = np.array(embedding).astype(
                                        "float32"
                                    )
                                    self.index.add(embedding_vector.reshape(1, -1))
                                    self.id_mapping[file_indices[i]] = texts[i]
                                texts = []
                                file_indices = []
                except Exception as e:
                    logger.error(f"Error processing {chunk_file}: {e}")
        # Process remaining texts
        if texts:
            embeddings = self.batch_get_embeddings(texts)
            for i, embedding in enumerate(embeddings):
                if not embedding:
                    continue
                embedding_cache[texts[i]] = embedding
                embedding_vector = np.array(embedding).astype("float32")
                self.index.add(embedding_vector.reshape(1, -1))
                self.id_mapping[file_indices[i]] = texts[i]

        # Save index and mappings
        faiss.write_index(self.index, index_path)
        with open(mapping_path, "wb") as f:
            pickle.dump(self.id_mapping, f)
        with open(self.cache_file, "w", encoding="utf-8") as cf:
            json.dump(embedding_cache, cf)
        logger.info("FAISS index built and saved successfully.")

    def create_semantic_chunks(self, text, max_tokens=800, overlap=200):
        """
        Advanced semantic chunking with more intelligent splitting
        """
        # Split by sections first, with more context preservation
        sections = text.split("\n\n")
        chunks = []
        current_chunk = []
        current_length = 0

        for section in sections:
            section = section.strip()
            if not section:
                continue

            # Split section into sentences
            sentences = section.split(". ")
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue

                # Estimate sentence length
                sentence_length = len(sentence.split())

                # More sophisticated chunk management
                if current_length + sentence_length > max_tokens and current_chunk:
                    # Join with spaces to preserve formatting
                    chunks.append(" ".join(current_chunk) + ".")

                    # Keep more context for overlap
                    current_chunk = (
                        current_chunk[-5:] if len(current_chunk) > 5 else current_chunk
                    )
                    current_length = sum(len(p.split()) for p in current_chunk)

                current_chunk.append(sentence)
                current_length += sentence_length

        if current_chunk:
            chunks.append(" ".join(current_chunk) + ".")

        logger.info(f"Created {len(chunks)} semantic chunks")
        return chunks

    def extract_topics(self, query: str) -> List[str]:
        """
        Improved topic extraction with more context-aware prompting.
        """
        system_prompt = f"""
        You are an expert at extracting key semantic topics from a query in Finnish.
        Extract 2-3 most important keywords that capture the semantic meaning.
        Focus on nouns and key concepts that represent the core of the query.
        """

        try:
            response = self.openai_client.chat.completions.create(model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query},
            ],
            max_tokens=30,
            temperature=0.2)
            topics_text = response.choices[0].message.content.strip()
            topics = [topic.strip() for topic in topics_text.split(",")]
            logger.info(f"Extracted topics: {topics}")
            return topics
        except Exception as e:
            logger.error(f"Error extracting topics: {str(e)}")
            return []

    def retrieve(self, query: str, top_k: int = 10) -> List[str]:
        """
        Enhanced retrieval with multiple strategies
        """
        try:
            # Generate primary query embedding with batch processing
            query_embedding = self.get_openai_embedding(query)

            # Optional: Generate topic-enhanced embedding
            topics = self.extract_topics(query)
            if topics:
                topic_embedding = self.get_openai_embedding(" ".join(topics))
                # Weighted combination of embeddings
                query_embedding = np.mean(
                    [np.array(query_embedding), np.array(topic_embedding)], axis=0
                )

            query_embedding = np.array(query_embedding).astype("float32").reshape(1, -1)

            # Perform similarity search
            D, I = self.index.search(query_embedding, top_k * 3)

            # Filter and rank results
            results = [self.id_mapping[idx] for idx in I[0] if idx in self.id_mapping]

            # Optional: Re-rank results using semantic similarity
            # disabled for now as it is slow andcauses a lot of openai api calls
            # results = sorted(
            #     results, key=lambda x: self._text_similarity(x, query), reverse=True
            # )

            logger.info(f"Retrieved {len(results)} chunks")
            return results[:top_k]


        except Exception as e:
            logger.error(f"Retrieval error: {str(e)}")
            return []

    def _text_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate cosine similarity between two texts using their embeddings
        """
        try:
            # Generate embeddings for both texts
            emb1 = self.get_openai_embedding(text1)
            emb2 = self.get_openai_embedding(text2)

            # Calculate cosine similarity
            emb1_norm = np.array(emb1) / np.linalg.norm(emb1)
            emb2_norm = np.array(emb2) / np.linalg.norm(emb2)

            similarity = np.dot(emb1_norm, emb2_norm)
            return similarity
        except Exception as e:
            logger.error(f"Similarity calculation error: {str(e)}")
            return 0.0

    def search_for_number(self, number: str) -> List[str]:
        """
        Directly search for text chunks containing a specific number.
        """
        results = []
        for idx, text in self.id_mapping.items():
            if number in text:
                results.append(text)
        return results

    def _find_section_header(self, text: str) -> str:
        """Try to find a relevant section header for the text"""
        # Look for common header patterns
        lines = text.split("\n")
        for line in lines[:2]:  # Check first couple of lines
            # Common header patterns in legal documents
            if any(
                pattern in line.lower() for pattern in ["§", "artikla", "kohta", "luku"]
            ):
                return line
            # Check for all-caps lines which are often headers
            if line.isupper() and len(line.split()) <= 10:
                return line
        return ""
