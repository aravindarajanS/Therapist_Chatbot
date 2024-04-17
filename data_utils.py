import pandas as pd


def load_data(data_path: str, max_rows: int = 15000) -> pd.DataFrame:
    """
    Loads a DataFrame from a CSV file path, limiting to a specified number of rows.

    Args:
        data_path (str): The path to the CSV file.
        max_rows (int, optional): The maximum number of rows to load. Defaults to 15000.

    Returns:
        pd.DataFrame: A DataFrame loaded from the specified path, limited to the provided rows.
    """
    return pd.read_csv(data_path, nrows=max_rows)


def create_processed_df(data: pd.DataFrame) -> pd.Series:
    """
    Preprocesses text in the provided DataFrame, adds prefixes for speaker identification,
    and creates a new combined text column.

    Args:
        data (pd.DataFrame): The DataFrame containing conversation data.

    Returns:
        pd.Series: A Series containing the processed combined text.
    """
    data["input"] = "Patient: " + data["input"]
    data["response"] = "Therapist: " + data["response"]
    return data["input"] + "\n" + data["response"]
