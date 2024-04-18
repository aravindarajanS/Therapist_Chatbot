import streamlit as st
from data_utils import load_data, create_processed_df
from embeddings import load_embeddings
from dialog_chain import create_chain, process_chat,create_db
from langchain_core.messages import AIMessage, HumanMessage
import time

data_path = "data/converse_data_processed_15k.csv"
embedding_path = "data/docs_embeddings.pkl"

# Load conversation data and embeddings
data = load_data(data_path)
processed_text = create_processed_df(data.copy())
embeddings = load_embeddings(embedding_path)

vectorStore = create_db(processed_text,embeddings)
# Create the dialogue chain
chain = create_chain(vectorstore=vectorStore)


st.title("Conversational Therapist Assistant")
st.subheader("Ask me anything!")
user_id = st.text_input("Enter your ID:",key='id')
if user_id:

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("Say something"):
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)

        if prompt=='exit':
            with open(f"data/output/{user_id}.txt", "w") as f:
                for message in st.session_state.messages:
                     f.write(f"{message}\n")

            exit()

        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        response = process_chat(chain, prompt, st.session_state.messages)
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(response)
        # Add assistant response to chat history

        st.session_state.messages.append({"role": "assistant", "content": response})
