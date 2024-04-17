import logging
import pickle
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)  # Set logging level


def compute_and_store_embeddings(
    model_name: str,  # Name of the pre-trained model
    data_path: str,  # Path to the CSV file
    output_file: str,  # Path to store the pickle file
    batch_size: int = 32,  # Batch size for MPS optimization
    device: str = "cpu"
) -> None:
    """
    Computes embeddings for input text in a CSV file using a pre-trained model,
    stores them in a pickle file, and creates a FAISS index for efficient retrieval.

    Args:
        model_name (str): Name of the pre-trained model (e.g., "togethercomputer/m2-bert-80M-2k-retrieval").
        data_path (str): Path to the CSV file containing an "input" column.
        output_file (str): Path to store the pickle file containing the computed embeddings.
        batch_size (int, optional): Batch size for input processing. Defaults to 32.
        device (str, optional): Device to use for computations ("cpu", "cuda", or "mps").
            Defaults to "mps" if available, otherwise "cuda".
    """

    model = AutoModelForSequenceClassification.from_pretrained(model_name, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model.to(device)

    # Load data from CSV file using pandas
    df = pd.read_csv(data_path)
    data = df["input"].tolist()  # Assuming "input" is the column name

    data=data[:15000]

    embeddings = []
    for i in range(0, len(data), batch_size):
        batch_texts = data[i: i + batch_size]
        encoded_inputs = tokenizer(batch_texts, padding="max_length", return_tensors="pt", truncation=True)
        encoded_inputs.to(device)

        with torch.no_grad():
            outputs = model(**encoded_inputs)
            batch_embeddings = outputs["sentence_embedding"].cpu().numpy()
            embeddings.extend(batch_embeddings.tolist())
            print(f'batch {i} : completed')

        # Log every 10000 embeddings
        if (i + 1) % 10000 == 0:
            logger.info(f"Processed {i + 1} input texts and generated embeddings.")

    # Store embeddings in a pickle file
    with open(output_file, "wb") as f:
        pickle.dump(embeddings, f)

    # Create a FAISS index for efficient retrieval (uncomment if needed)
    # d = len(embeddings[0])  # Assuming all embeddings have the same dimension
    # index = faiss.IndexFlatL2(d)
    # index.add(np.asarray(embeddings))
    # # Save the FAISS index to a file (optional)
    # faiss.write_index(index, "my_faiss_index.faiss")


# Example usage
model_name = "togethercomputer/m2-bert-80M-2k-retrieval"
data_path = "data/converse_data_processed.csv"
output_file = "data/docs_embeddings.pkl"
compute_and_store_embeddings(model_name, data_path, output_file)

# To load the embeddings later:
# with open(output_file, "rb") as f:
#     loaded_embeddings = pickle.load(f)
