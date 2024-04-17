import os
import pandas as pd


def load_data(data_path: str) -> pd.DataFrame:
  """
  Loads a DataFrame from a Parquet file path.

  Args:
      data_path: The path to the Parquet file.

  Returns:
      A pandas DataFrame loaded from the specified path.
  """
  return pd.read_parquet(data_path)


def preprocess_text(text: str, marker: str = "[INST]") -> str:
  """
  Preprocesses text by replacing the closing marker with the opening marker.

  Args:
      text: The text string to preprocess.
      marker: The marker string to replace (defaults to "[INST]").

  Returns:
      The preprocessed text string.
  """
  return text.replace(f"[/{marker}]", marker)


def process_conversation(conversation: str, remove_text: str = "") -> str:
  """
  Processes a conversation by splitting based on the marker, removing unwanted text,
  and cleaning leading/trailing whitespace and special characters.

  Args:
      conversation: The conversation string to process.
      remove_text: Optional text string to remove (defaults to "").

  Returns:
      The processed conversation string.
  """
  cleaned_segments = []
  for segment in conversation.split("[INST]"):
    cleaned_segment = segment.strip()
    if cleaned_segment:
      cleaned_segment = cleaned_segment.replace(remove_text, "").replace("</s><s>", "").replace("<s>", "").replace("</s>", "")
      cleaned_segments.append(cleaned_segment)
  return " ".join(cleaned_segments)


def create_conversation_pairs(processed_conversations: list) -> list:
  """
  Creates conversation pairs (input, response) from a list of processed conversations.

  Args:
      processed_conversations: A list of processed conversation strings.

  Returns:
      A list of tuples containing conversation pairs (input, response).
  """
  conversation_pairs = []
  for i in range(0, len(processed_conversations), 2):
    if i + 1 < len(processed_conversations):
      conversation_pairs.append((processed_conversations[i], processed_conversations[i + 1]))
  return conversation_pairs


def main():
  """
  The main function that loads data, performs preprocessing, and creates conversation pairs.
  """
  data_path = "converse_data.parquet"
  output_path = "converse_data_processed.csv"

  text_to_remove="""<<SYS>>
You are a helpful and joyous mental therapy assistant. Always answer as helpfully and cheerfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content.Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
<</SYS>
"""

  df = load_data(data_path)
  text_list = df["text"].tolist()

  # Preprocess text to ensure consistent markers
  modified_texts = [preprocess_text(text) for text in text_list]

  processed_conversations = []
  for conversation in modified_texts:
    processed_conversation = process_conversation(conversation, remove_text=text_to_remove)
    processed_conversations.append(processed_conversation)

  conversation_pairs = create_conversation_pairs(processed_conversations)

  # Create DataFrame and save to CSV
  df = pd.DataFrame(conversation_pairs, columns=["input", "response"])
  df.to_csv(output_path, index=False)


if __name__ == "__main__":
  main()
