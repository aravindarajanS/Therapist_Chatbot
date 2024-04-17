import streamlit as st
from data_utils import load_data, create_processed_df
from embeddings import load_embeddings
from dialog_chain import create_chain, process_chat,create_db
from langchain_core.messages import AIMessage, HumanMessage
import time


data_path = "data/converse_data_processed.csv"
embedding_path = "data/docs_embeddings.pkl"


# Load conversation data and embeddings
data = load_data(data_path)
processed_text = create_processed_df(data.copy())
embeddings = load_embeddings(embedding_path)

vectorStore = create_db(processed_text,embeddings)
# Create the dialogue chain
chain = create_chain(vectorstore=vectorStore)

# Initialize chat history
chat_history = []

st.title("Conversational Therapist Assistant")
st.subheader("Ask me anything!")

user_id = st.text_input("Enter your ID:")

if user_id:
    chat_history = []

    input_index = 0

    while True:
        user_input = st.text_input("You:", key=f"user_input_{input_index}")
        input_index += 1

        if user_input.lower() == "exit":
            break


        try:
            response = process_chat(chain, user_input, chat_history)
            chat_history.append(HumanMessage(content=user_input))
            chat_history.append(AIMessage(content=response))

            st.write(f"Assistant: {response}")

        except Exception as e:
            st.error(f"An error occurred: {e}")


        user_input = ""  

        time.sleep(5)

        # Save chat history to a text file
    with open(f"data/output/{user_id}.txt", "w") as f:
        for message in chat_history:
             f.write(f"{message}\n")
