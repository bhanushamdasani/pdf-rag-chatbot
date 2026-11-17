"""Vector store module: creates and manages FAISS vector index with Google Gemini or Local embeddings."""

from typing import List, Optional
import os
import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from sklearn.feature_extraction.text import TfidfVectorizer


class LocalTFIDFEmbeddings(Embeddings):
    """
    Lightweight local embeddings using TF-IDF for offline demonstration and testing.
    Requires $0 and no API keys or internet connection.
    """
    def __init__(self, dim: int = 128):
        self.dim = dim
        self.vectorizer = TfidfVectorizer(max_features=dim, stop_words="english")
        self.fitted = False

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        matrix = self.vectorizer.fit_transform(texts).toarray()
        if matrix.shape[1] < self.dim:
            pad = np.zeros((matrix.shape[0], self.dim - matrix.shape[1]))
            matrix = np.hstack([matrix, pad])
        self.fitted = True
        return matrix.tolist()

    def embed_query(self, text: str) -> List[float]:
        if not self.fitted:
            return [0.0] * self.dim
        vec = self.vectorizer.transform([text]).toarray()
        if vec.shape[1] < self.dim:
            pad = np.zeros((vec.shape[0], self.dim - vec.shape[1]))
            vec = np.hstack([vec, pad])
        return vec[0].tolist()


def get_embedding_model(
    google_api_key: Optional[str] = None,
    model_name: str = "models/gemini-embedding-001",
    demo_mode: bool = False
) -> Embeddings:
    """
    Initializes embeddings.
    If demo_mode is True, uses LocalTFIDFEmbeddings (100% offline, $0).
    Otherwise, uses GoogleGenerativeAIEmbeddings.

    Args:
        google_api_key: Optional Google Gemini API key
        model_name: Gemini embedding model name
        demo_mode: When True, uses local offline embeddings

    Returns:
        Embeddings instance
    """
    if demo_mode:
        return LocalTFIDFEmbeddings()

    if not google_api_key:
        raise ValueError("Google API Key must be provided for Gemini Embeddings.")

    return GoogleGenerativeAIEmbeddings(
        model=model_name,
        google_api_key=google_api_key,
        task_type="retrieval_document"
    )


def create_faiss_vector_store(
    chunks: List[Document],
    embeddings: Embeddings
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
    embeddings: Embeddings
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
