# retrieval/retrieval_system.py
import os
import faiss
import pickle
import openai
import numpy as np
import logging
import time
import json
from typing import List

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
        Initializes the RetrievalSystem.

        Args:
            chunk_dir (str): Directory containing text chunk files.
            index_path (str): Path to save/load the FAISS index.
            mapping_path (str): Path to save/load the ID to text mapping.
            cache_file (str): Path to the embedding cache file.
        """
        self.chunk_dir = chunk_dir
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set.")
        openai.api_key = self.openai_api_key
        self.dimension = 1536  # OpenAI's text-embedding-ada-002 has 1536 dimensions
        self.cache_file = cache_file

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

    def get_openai_embedding(
        self, text: str, max_retries: int = 5, backoff_factor: float = 0.5
    ) -> List[float]:
        """
        Fetches the embedding for the given text using OpenAI's API with retry logic.

        Args:
            text (str): The text to embed.
            max_retries (int): Maximum number of retries.
            backoff_factor (float): Factor for exponential backoff.

        Returns:
            List[float]: The embedding vector.
        """
        for attempt in range(max_retries):
            try:
                response = openai.Embedding.create(
                    input=text, model="text-embedding-ada-002"
                )
                embedding = response["data"][0]["embedding"]
                return embedding
            except openai.error.RateLimitError:
                wait_time = backoff_factor * (2**attempt)
                logger.warning(
                    f"Rate limit exceeded. Retrying in {wait_time} seconds..."
                )
                time.sleep(wait_time)
            except openai.error.OpenAIError as e:
                logger.error(f"OpenAI API error: {e}")
                break  # Non-retriable error
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                break  # Non-retriable error
        return []

    def get_openai_embeddings(
        self, texts: List[str], max_retries: int = 5, backoff_factor: float = 0.5
    ) -> List[List[float]]:
        """
        Fetches embeddings for a list of texts using OpenAI's API with retry logic.

        Args:
            texts (List[str]): List of texts to embed.
            max_retries (int): Maximum number of retries.
            backoff_factor (float): Factor for exponential backoff.

        Returns:
            List[List[float]]: List of embedding vectors.
        """
        for attempt in range(max_retries):
            try:
                response = openai.Embedding.create(
                    input=texts, model="text-embedding-ada-002"
                )
                embeddings = [data["embedding"] for data in response["data"]]
                return embeddings
            except openai.error.RateLimitError:
                wait_time = backoff_factor * (2**attempt)
                logger.warning(
                    f"Rate limit exceeded. Retrying in {wait_time} seconds..."
                )
                time.sleep(wait_time)
            except openai.error.OpenAIError as e:
                logger.error(f"OpenAI API error: {e}")
                break  # Non-retriable error
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                break  # Non-retriable error
        return []

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
                                embeddings = self.get_openai_embeddings(texts)
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
            embeddings = self.get_openai_embeddings(texts)
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

    def create_semantic_chunks(self, text, max_tokens=512, overlap=50):
        """
        Creates semantic chunks based on sentence boundaries and optional overlapping.
        """
        sentences = text.split(". ")
        chunks = []
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            sentence_length = len(sentence.split())
            if current_length + sentence_length <= max_tokens:
                current_chunk.append(sentence)
                current_length += sentence_length
            else:
                chunks.append(". ".join(current_chunk).strip())
                # Create overlap with last few sentences
                overlap_start = max(0, len(current_chunk) - overlap)
                current_chunk = current_chunk[overlap_start:] + [sentence]
                current_length = sum(len(s.split()) for s in current_chunk)

        if current_chunk:
            chunks.append(". ".join(current_chunk).strip())

        return chunks

    def retrieve(self, query: str, top_k: int = 5) -> List[str]:
        """
        Retrieves the top_k most similar text chunks to the query.

        Args:
            query (str): The user query.
            top_k (int): Number of top similar chunks to retrieve.

        Returns:
            List[str]: List of retrieved text chunks.
        """
        try:
            logger.info(f"Generating embedding for query: {query}")
            query_embedding = self.get_openai_embedding(query)
            if not query_embedding:
                logger.error("Failed to generate embedding for the query.")
                return []
            query_vector = np.array(query_embedding).astype("float32").reshape(1, -1)
            distances, indices = self.index.search(query_vector, top_k)
            results = []
            for idx in indices[0]:
                if idx == -1:
                    continue  # FAISS returns -1 if no more neighbors are found
                text = self.id_mapping.get(idx, "").decode("unicode-escape")
                if text:
                    results.append(text)
            logger.info(f"Retrieved {len(results)} relevant chunks.")
            return results
        except Exception as e:
            logger.error(f"Error during retrieval: {e}")
            return []
