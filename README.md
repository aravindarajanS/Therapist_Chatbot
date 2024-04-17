# Conversational Therapist Assistant

This project is a conversational therapist assistant built using langchain, huggingface and FAISS. It provides a simple interface for users to engage in a conversation with the assistant.

## Features

- Users can interact with the assistant by typing messages.
- The assistant provides responses based on the conversation history.
- Conversation history is saved to a text file for each user session.

## Installation

    ```bash
    pip install -r requirements.txt
    ```

## Usage

 streamlit app is not functional as expected some errors has to be fixed, the stremlit code here is just for demonstration purpose

 ## Data

the data used in this app is from following link
https://huggingface.co/datasets/vibhorag101/phr_mental_therapy_dataset?row=9

The above data seems to be more suitable for the task, data is processed and stored in a seperate file.


## File Structure

- `streamlit.py`: Main script containing the Streamlit app.
- `data/`: Directory containing conversation data and embeddings.
- `data_preparation.py`: script for processing data
- `text_encoder.py`: script for generating embeddings for text
- `data_utils.py`: Module for loading and processing data.
- `embeddings.py`: Module for loading embeddings.
- `dialog_chain.py`: Module for creating and processing the dialogue chain.
- `main.py`: main script of the chatbot
- `evaluation.py`: evaluation script consist of faithfulness, correctness and MRR metrics
- `README.md`: This file.
