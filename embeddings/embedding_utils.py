import openai
import numpy as np
from typing import List

# Initialize OpenAI API key
openai.api_key = "YOUR_OPENAI_API_KEY"


def get_embedding(text: str, model: str = "text-embedding-ada-002") -> List[float]:
    """
    Generates an embedding for the given text using OpenAI's API.

    Args:
        text (str): The input text to embed.
        model (str): The OpenAI model to use for embedding.

    Returns:
        List[float]: The embedding vector.
    """
    response = openai.Embedding.create(input=text, model=model)
    embedding = response["data"][0]["embedding"]
    return embedding


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    """
    Normalizes a vector to have unit length.

    Args:
        vector (np.ndarray): The input vector.

    Returns:
        np.ndarray: The normalized vector.
    """
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm
