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
import torch
import torch.nn.functional as F

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

    def create_semantic_chunks(self, text, max_tokens=800, overlap=200):
        """
        Creates semantic chunks with larger size and overlap to maintain context.
        """
        # Split by sections first
        sections = text.split("\n\n")
        chunks = []
        current_chunk = []
        current_length = 0

        for section in sections:
            section = section.strip()
            if not section:
                continue

            # Estimate section length
            section_length = len(section.split())

            if current_length + section_length > max_tokens and current_chunk:
                # Join with double newlines to preserve formatting
                chunks.append("\n\n".join(current_chunk))
                # Keep significant overlap for context
                current_chunk = (
                    current_chunk[-2:] if len(current_chunk) > 2 else current_chunk[-1:]
                )
                current_length = sum(len(p.split()) for p in current_chunk)

            current_chunk.append(section)
            current_length += section_length

        if current_chunk:
            chunks.append("\n\n".join(current_chunk))

        return chunks

    def search_for_number(self, number: str) -> List[str]:
        """
        Directly search for text chunks containing a specific number.
        """
        results = []
        for idx, text in self.id_mapping.items():
            if number in text:
                results.append(text)
        return results

    def extract_topics(self, query: str) -> List[str]:
        """
        Use OpenAI to extract potential topics from the query.
        """
        prompt = f"{query}"
        system_prompt = "Extract key topics from the query and respond with a comma-separated list of single words in the same language as the query."

        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=50,
                temperature=0.3,
            )
            topics_text = response.choices[0].message["content"].strip()
            # Assume topics are returned as a comma-separated list
            topics = [topic.strip() for topic in topics_text.split(",")]
            print(f"topics: {topics}")
            return topics
        except Exception as e:
            logger.error(f"Error extracting topics with OpenAI: {str(e)}")
            return []

    def retrieve(self, query: str, top_k: int = 10) -> List[str]:
        """
        Retrieve the most relevant text chunks for a given query with improved context.
        """
        try:

            logger.info(f"Generating embedding for query: {query}")

            # Use OpenAI to extract topics from the query
            topic_results = []
            topic_results.extend(self.extract_topics(query))

            topic_results_str = " ".join(topic_results)

            # Generate embedding for the query
            query_embedding = self.get_openai_embedding(f"{query} {topic_results_str}")
            if not query_embedding:
                logger.error("Failed to generate embedding for query")
                return topic_results  # Return topic results if embedding fails

            # Convert embedding to correct format for FAISS
            query_embedding = np.array(query_embedding).astype("float32").reshape(1, -1)

            # Search the index with more neighbors to filter
            D, I = self.index.search(query_embedding, top_k * 3)

            # Get the corresponding texts and sort by relevance score
            results = []
            seen_content = set()  # To avoid duplicate content

            for score, idx in zip(D[0], I[0]):
                if idx in self.id_mapping:
                    text = self.id_mapping[idx]

                    # Skip if too similar to already included content
                    text_normalized = " ".join(text.split())  # Normalize whitespace
                    if any(
                        self._text_similarity(text_normalized, seen) > 0.8
                        for seen in seen_content
                    ):
                        continue

                    results.append(
                        {
                            "text": text,
                            "score": float(score),  # Convert to float for sorting
                        }
                    )
                    seen_content.add(text_normalized)

            # Combine number, topic, and embedding results
            combined_results = []
            # Check if the query contains a number
            import re

            number_match = re.search(r"\d+", query)
            if number_match:
                number = number_match.group(0)
                number_results = self.search_for_number(number)
                combined_results.extend(number_results)

            combined_results.extend([r["text"] for r in results[:top_k]])

            return combined_results

        except Exception as e:
            logger.error(f"Error during retrieval: {str(e)}")
            return []

    def _text_similarity(self, text1: str, text2: str) -> float:
        """Calculates cosine similarity between two texts using their embeddings."""
        embedding1 = self.get_openai_embedding(text1)
        embedding2 = self.get_openai_embedding(text2)

        if not embedding1 or not embedding2:
            return 0.0

        embedding1 = torch.tensor(embedding1)
        embedding2 = torch.tensor(embedding2)

        similarity = F.cosine_similarity(embedding1, embedding2, dim=0)
        return similarity.item()

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
