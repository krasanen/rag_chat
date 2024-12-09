import os
from retrieval.retrieval_system import RetrievalSystem


def load_data(file_path: str) -> list:
    """
    Loads and splits data into chunks.

    Args:
        file_path (str): Path to the data file.

    Returns:
        list: List of text chunks.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data = f.read()
    # Simple split by paragraphs. Adjust as needed.
    chunks = data.split("\n\n")
    return [chunk.strip() for chunk in chunks if chunk.strip()]


def main():
    data_file = "source_txts/tietoala_tes_suomi_2023_web.txt"
    texts = load_data(data_file)

    if not texts:
        print("No texts found in the data file.")
        return

    # Initialize Retrieval System with embedding dimension (e.g., 1536 for text-embedding-ada-002)
    retrieval_system = RetrievalSystem(dimension=1536)

    # Add texts to the index
    retrieval_system.add_texts(texts)

    # Example query
    query = "Your example question here."
    results = retrieval_system.retrieve(query, top_k=3)
    print("Top 3 results:")
    for i, res in enumerate(results, 1):
        print(f"{i}. {res}")


if __name__ == "__main__":
    main()
