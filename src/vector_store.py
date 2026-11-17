"""Vector store module: creates and manages FAISS vector index with Google Gemini embeddings."""

from typing import List, Optional
import os
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings


def get_embedding_model(
    google_api_key: str,
    model_name: str = "models/embedding-001"
) -> GoogleGenerativeAIEmbeddings:
    """
    Initializes Google Gemini Embeddings.

    Args:
        google_api_key: Google Gemini API key
        model_name: Embedding model (e.g. 'models/embedding-001' or 'models/text-embedding-004')

    Returns:
        GoogleGenerativeAIEmbeddings instance
    """
    if not google_api_key:
        raise ValueError("Google API Key must be provided for Gemini Embeddings.")

    return GoogleGenerativeAIEmbeddings(
        model=model_name,
        google_api_key=google_api_key,
        task_type="retrieval_document"
    )


def create_faiss_vector_store(
    chunks: List[Document],
    embeddings: GoogleGenerativeAIEmbeddings
) -> FAISS:
    """
    Builds a FAISS in-memory vector store from document chunks.

    Args:
        chunks: List of chunked Document objects
        embeddings: Embeddings model instance

    Returns:
        FAISS vector store instance
    """
    if not chunks:
        raise ValueError("Cannot create vector store from empty document list.")

    vector_store = FAISS.from_documents(documents=chunks, embedding=embeddings)
    return vector_store


def save_vector_store(vector_store: FAISS, folder_path: str = "faiss_index") -> None:
    """Saves FAISS index to local disk."""
    os.makedirs(folder_path, exist_ok=True)
    vector_store.save_local(folder_path)


def load_vector_store(
    folder_path: str,
    embeddings: GoogleGenerativeAIEmbeddings
) -> Optional[FAISS]:
    """Loads FAISS index from local disk."""
    if not os.path.exists(folder_path):
        return None
    return FAISS.load_local(
        folder_path,
        embeddings,
        allow_dangerous_deserialization=True
    )


def get_retriever(vector_store: FAISS, top_k: int = 4):
    """
    Configures similarity search retriever.

    Args:
        vector_store: FAISS vector store instance
        top_k: Number of most relevant chunks to return per query

    Returns:
        VectorStoreRetriever
    """
    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k}
    )

