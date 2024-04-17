from data_utils import load_data, create_processed_df
from embeddings import load_embeddings
from dialog_chain import create_chain, process_chat,create_db
from langchain_core.messages import AIMessage, HumanMessage


if __name__ == "__main__":

    data_path = "data/converse_data_processed.csv"
    embedding_path = "data/docs_embeddings.pkl"
    data = load_data(data_path)
    processed_text = create_processed_df(data.copy())
    embeddings = load_embeddings(embedding_path)
    
    vectorStore = create_db(processed_text,embeddings)
    chain = create_chain(vectorstore=vectorStore)
    
    # Initialize chat history
    chat_history = []
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break
        response = process_chat(chain, user_input, chat_history)
        chat_history.append(HumanMessage(content=user_input))
        chat_history.append(AIMessage(content=response))
        print("Assistant:", response)
