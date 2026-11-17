"""RAG Chain module: builds conversational QA chain with memory and citation tracking."""

from typing import Dict, Any, List
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.vectorstores import VectorStoreRetriever

# Custom grounded prompt template
QA_SYSTEM_PROMPT = """You are an intelligent, accurate document assistant.
Your task is to answer user questions based ONLY on the provided document excerpts.

Guidelines:
1. Ground your answers strictly in the given context.
2. If the answer cannot be determined from the context, state clearly: "I cannot find the answer to this question in the uploaded document."
3. Do not make up facts or extrapolate beyond what is stated.
4. Keep answers concise, clear, and well-structured with bullet points when applicable.

Context:
{context}

Question:
{question}

Helpful Answer:"""


def get_gemini_llm(
    google_api_key: str,
    model_name: str = "gemini-1.5-flash",
    temperature: float = 0.2
) -> ChatGoogleGenerativeAI:
    """
    Initializes Google Gemini Chat LLM.

    Args:
        google_api_key: Google Gemini API key
        model_name: Gemini model variant ('gemini-1.5-flash', 'gemini-1.5-pro', etc.)
        temperature: Sampling temperature (lower = more deterministic/factual)

    Returns:
        ChatGoogleGenerativeAI instance
    """
    if not google_api_key:
        raise ValueError("Google API Key must be provided for Gemini LLM.")

    return ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=google_api_key,
        temperature=temperature,
        convert_system_message_to_human=True
    )


def build_conversational_rag_chain(
    llm: Any,
    retriever: VectorStoreRetriever,
    window_k: int = 5
) -> ConversationalRetrievalChain:
    """
    Builds a ConversationalRetrievalChain with sliding window memory and source return.

    Args:
        llm: Language model instance
        retriever: VectorStoreRetriever instance
        window_k: Number of conversation turns to preserve in memory

    Returns:
        Configured ConversationalRetrievalChain
    """
    memory = ConversationBufferWindowMemory(
        k=window_k,
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )

    qa_prompt = PromptTemplate(
        template=QA_SYSTEM_PROMPT,
        input_variables=["context", "question"]
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": qa_prompt},
        return_source_documents=True,
        verbose=False
    )

    return chain


def format_citations(source_documents: List[Any]) -> List[Dict[str, Any]]:
    """
    Formats retrieved documents into clean citation metadata.

    Args:
        source_documents: List of retrieved Document chunks

    Returns:
        List of dicts containing page number, source, and text excerpt
    """
    citations = []
    seen = set()

    for doc in source_documents:
        page = doc.metadata.get("page", 1)
        source = doc.metadata.get("source", "Document")
        snippet = doc.page_content.strip()
        # Truncate snippet preview
        preview = snippet if len(snippet) <= 220 else snippet[:220] + "..."

        key = (page, preview)
        if key not in seen:
            seen.add(key)
            citations.append({
                "page": page,
                "source": source,
                "preview": preview,
                "full_text": snippet
            })

    return citations

