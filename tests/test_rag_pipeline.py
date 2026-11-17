"""Unit tests verifying document chunking, FAISS vector indexing, memory, and chain logic."""

import pytest
from langchain_core.documents import Document
from langchain_community.embeddings import FakeEmbeddings
from langchain_community.llms.fake import FakeListLLM

from src.document_processor import chunk_documents
from src.vector_store import create_faiss_vector_store, get_retriever, LocalTFIDFEmbeddings
from src.rag_chain import build_conversational_rag_chain, format_citations, DemoExtractiveLLM


def test_chunk_documents():
    """Verify document chunking retains metadata and chunks properly."""
    sample_text = "Machine learning is a field of artificial intelligence. " * 30
    doc = Document(
        page_content=sample_text,
        metadata={"source": "test.pdf", "page": 1}
    )

    chunks = chunk_documents([doc], chunk_size=200, chunk_overlap=40)

    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.metadata["source"] == "test.pdf"
        assert chunk.metadata["page"] == 1
        assert "chunk_id" in chunk.metadata
        assert len(chunk.page_content) <= 250


def test_faiss_vector_store_and_retriever():
    """Verify FAISS vector index creation and retrieval using semantic TF-IDF embeddings."""
    docs = [
        Document(page_content="Google Gemini is a multimodal AI model.", metadata={"page": 1}),
        Document(page_content="LangChain is an orchestration framework for LLMs.", metadata={"page": 2}),
        Document(page_content="FAISS provides efficient similarity search for dense vectors.", metadata={"page": 3}),
    ]

    embeddings = LocalTFIDFEmbeddings(dim=32)
    vector_store = create_faiss_vector_store(docs, embeddings)
    retriever = get_retriever(vector_store, top_k=2)

    results = retriever.invoke("What is LangChain?")
    assert len(results) == 2
    assert any("LangChain" in r.page_content for r in results)


def test_conversational_rag_chain_with_memory():
    """Verify conversational chain execution, memory tracking, and source citations."""
    docs = [
        Document(page_content="Project Alpha was launched in 2024 to accelerate AI research.", metadata={"page": 5, "source": "alpha.pdf"}),
        Document(page_content="Project Alpha is funded with a 10 million dollar grant.", metadata={"page": 6, "source": "alpha.pdf"}),
    ]

    embeddings = FakeEmbeddings(size=32)
    vector_store = create_faiss_vector_store(docs, embeddings)
    retriever = get_retriever(vector_store, top_k=1)

    # Note: On turn 2 with chat history, ConversationalRetrievalChain calls LLM twice:
    # 1. Question generator (condensing question with memory)
    # 2. QA answer generation
    fake_responses = [
        "Project Alpha launched in 2024.",                               # Turn 1 answer
        "Standalone question: How much funding did Project Alpha get?",    # Turn 2 question condensing
        "It received a 10 million dollar grant."                          # Turn 2 answer
    ]
    llm = FakeListLLM(responses=fake_responses)

    chain = build_conversational_rag_chain(llm=llm, retriever=retriever, window_k=3)

    # First turn
    result_1 = chain.invoke({"question": "When was Project Alpha launched?"})
    assert result_1["answer"] == "Project Alpha launched in 2024."
    assert len(result_1["source_documents"]) >= 1

    # Check citations formatter
    citations = format_citations(result_1["source_documents"])
    assert len(citations) >= 1
    assert citations[0]["page"] in [5, 6]
    assert "alpha.pdf" in citations[0]["source"]

    # Second turn (memory context in action)
    result_2 = chain.invoke({"question": "How much funding did it get?"})
    assert result_2["answer"] == "It received a 10 million dollar grant."
    assert len(result_2["source_documents"]) >= 1


def test_offline_demo_pipeline():
    """Verify offline demonstration mode using LocalTFIDFEmbeddings and DemoExtractiveLLM."""
    docs = [
        Document(page_content="Enterprise RAG adoption reached 78 percent in 2025.", metadata={"page": 1, "source": "report.pdf"}),
        Document(page_content="Project Titan was granted 15.5 million dollars in funding.", metadata={"page": 2, "source": "report.pdf"}),
        Document(page_content="FAISS vector similarity search averaged 14.2 milliseconds latency.", metadata={"page": 3, "source": "report.pdf"}),
    ]

    embeddings = LocalTFIDFEmbeddings(dim=32)
    vector_store = create_faiss_vector_store(docs, embeddings)
    retriever = get_retriever(vector_store, top_k=1)
    llm = DemoExtractiveLLM()

    chain = build_conversational_rag_chain(llm=llm, retriever=retriever, window_k=2)
    result = chain.invoke({"question": "What was the adoption rate in 2025?"})

    assert "78 percent" in result["answer"]
    assert len(result["source_documents"]) == 1
    assert result["source_documents"][0].metadata["page"] == 1


def test_empty_document_handling():
    """Ensure vector store raises ValueError when given empty chunks."""
    embeddings = FakeEmbeddings(size=32)
    with pytest.raises(ValueError):
        create_faiss_vector_store([], embeddings)
