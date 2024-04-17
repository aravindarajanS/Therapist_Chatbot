import numpy as np
import pickle
import faiss


def load_embeddings(embedding_path: str) -> np.ndarray:
    """
    Loads embeddings from a pickle file.

    Args:
        embedding_path (str): The path to the pickle file containing embeddings.

    Returns:
        np.ndarray: A NumPy array containing the loaded embeddings,
            or None if an error occurs.
    """
    try:
        with open(embedding_path, "rb") as f:
            embeddings = pickle.load(f)
        return np.array(embeddings)
    except (FileNotFoundError, pickle.UnpicklingError) as e:
        print(f"Error loading embeddings: {e}")
        return None


def create_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatL2:
    """
    Creates a FAISS index for efficient similarity search on the embeddings.

    Args:
        embeddings (np.ndarray): A NumPy array containing the embeddings.

    Returns:
        faiss.IndexFlatL2: A FAISS index object for searching the embeddings.
    """
    index = faiss.IndexFlatL2(768)
    index.add(embeddings)
    return index
