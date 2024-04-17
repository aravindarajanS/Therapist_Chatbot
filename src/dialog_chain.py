from typing import  Union, List
import numpy as np
from langchain import llms
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage, HumanMessage
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import MessagesPlaceholder
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain_community.vectorstores.faiss import FAISS


def create_db(docs: List[str], embeddings: np.ndarray):
    """
    Creates a FAISS vector store from document-embedding pairs.

    Args:
        docs (List[str]): A list of documents.
        embeddings (np.ndarray): A NumPy array containing the embeddings for the documents.
            The shape of the array should be (num_documents, embedding_dim).
        embed_function (Callable[[str], np.ndarray]): A function that takes a document as input
            and returns its embedding as a NumPy array.

    Returns:
        FAISSVectorStore: A FAISS vector store containing the document embeddings.
    """
    
    model_name = "togethercomputer/m2-bert-80M-2k-retrieval"
    model_kwargs = {'device': 'cpu','trust_remote_code':True}
    encode_kwargs = {'normalize_embeddings': False}
    hf = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

    doc_embedding_pairs = zip(docs, embeddings)
    vector_store = FAISS.from_embeddings(doc_embedding_pairs, hf)
    return vector_store

def create_chain(vectorstore):
    """
    Creates a LangChain dialogue chain using a retrieval system or a simple LLM call.

    Args:
        vectorstore (Union[FAISS.FAISSVectorStore, None], optional):
            The vector store to use for retrieval, or None for a direct LLM call.
            Defaults to None.

    Returns:
        chains.RetrievalChain: The created dialogue LangChain.
        
        
    """
    


    model = llms.Ollama(model="gemma:7b")

    prompt = ChatPromptTemplate.from_messages([
         ("system",
            "You are a helpful and joyous mental therapy assistant. Always answer as helpfully and cheerfully as possible, while being safe.\n"
            "Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content.\n"
            "Please ensure that your responses are socially unbiased and positive in nature.\n"
            "If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct.\n"
            "If you don't know the answer to a question, please don't share false information.\n"
            "\n"
            "Below are some relevant conversations between patient and the therapist for your reference\n"
            "context: {context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}")
    ])

     
    chain = create_stuff_documents_chain(
        llm=model,
        prompt=prompt
    )
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 1})

    retriever_prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])

    history_aware_retriever = create_history_aware_retriever(
        llm=model,
        retriever=retriever,
        prompt=retriever_prompt
    )

    
    retrieval_chain = create_retrieval_chain(
       
        history_aware_retriever,
        chain
    )

    return retrieval_chain


def process_chat(chain, question: str, chat_history: List[Union[HumanMessage, AIMessage]]) -> str:
    """
    Processes a new user question by invoking the dialogue LangChain.

    Args:
        chain (chains.RetrievalChain): The retrieval chain for dialogue generation.
        question (str): The user's question.
        chat_history (List[Union[HumanMessage, AIMessage]]): The conversation history.

    Returns:
        str: The generated response to the user's question.
    """
    response = chain.invoke({
        "chat_history": chat_history,
        "input": question,
    })
    return response["answer"]

