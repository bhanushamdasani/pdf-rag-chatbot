"""PDF Chatbot using LangChain, FAISS, and Google Gemini.
Production-ready Streamlit application with conversational memory and source citations.
"""

import os
import streamlit as st
from dotenv import load_dotenv

from src.document_processor import extract_text_from_pdf, chunk_documents
from src.vector_store import get_embedding_model, create_faiss_vector_store, get_retriever
from src.rag_chain import get_gemini_llm, build_conversational_rag_chain, format_citations

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="PDF Chatbot | LangChain + Gemini RAG",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.1rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .badge-container {
        display: flex;
        gap: 8px;
        margin-bottom: 1.5rem;
        flex-wrap: wrap;
    }
    .tech-badge {
        background-color: #EEF2FF;
        color: #4F46E5;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 600;
        border: 1px solid #E0E7FF;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 12px;
        margin-top: 10px;
    }
    .source-box {
        background-color: #F1F5F9;
        border-left: 3px solid #4F46E5;
        padding: 8px 12px;
        border-radius: 4px;
        margin-top: 6px;
        font-size: 0.85rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chain" not in st.session_state:
    st.session_state.chain = None

if "doc_info" not in st.session_state:
    st.session_state.doc_info = None

if "indexed" not in st.session_state:
    st.session_state.indexed = False


# Sidebar Configuration
with st.sidebar:
    st.title("⚙️ Configuration")
    
    # API Key Input
    env_key = os.getenv("GOOGLE_API_KEY", "")
    api_key = st.text_input(
        "Google Gemini API Key",
        value=env_key,
        type="password",
        help="Get your free API key at https://aistudio.google.com/app/apikey"
    )

    if not api_key:
        st.info("💡 **Free API Key**: Google AI Studio provides a free tier. [Click here to create a key](https://aistudio.google.com/app/apikey).")

    st.divider()

    # Model Parameters
    st.subheader("🤖 Model Settings")
    model_choice = st.selectbox(
        "Gemini Model",
        options=["gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.0-flash"],
        index=0,
        help="gemini-1.5-flash is fast, accurate, and free-tier friendly."
    )

    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.2,
        step=0.05,
        help="Lower values yield more factual and grounded answers."
    )

    top_k = st.slider(
        "Top-k Retrievable Chunks",
        min_value=1,
        max_value=8,
        value=4,
        help="Number of document chunks passed to the LLM per query."
    )

    st.divider()

    # Document Upload Section
    st.subheader("📑 Document Upload")
    uploaded_file = st.file_uploader(
        "Upload a PDF",
        type=["pdf"],
        help="Upload any PDF to build the FAISS vector index."
    )

    chunk_size = st.number_input("Chunk Size (characters)", min_value=300, max_value=3000, value=1000, step=100)
    chunk_overlap = st.number_input("Chunk Overlap (characters)", min_value=0, max_value=500, value=200, step=50)

    if uploaded_file and st.button("🚀 Process & Index PDF", use_container_width=True):
        if not api_key:
            st.error("Please enter your Google Gemini API key above to proceed.")
        else:
            with st.spinner("Extracting text and chunking document..."):
                try:
                    # 1. Extract text
                    raw_docs = extract_text_from_pdf(uploaded_file, filename=uploaded_file.name)
                    if not raw_docs:
                        st.error("No extractable text found in the uploaded PDF.")
                    else:
                        # 2. Chunk documents
                        chunks = chunk_documents(raw_docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

                        # 3. Embed & build FAISS vector store
                        with st.spinner("Generating Gemini embeddings and building FAISS vector index..."):
                            embeddings = get_embedding_model(google_api_key=api_key)
                            vector_store = create_faiss_vector_store(chunks, embeddings)
                            retriever = get_retriever(vector_store, top_k=top_k)

                        # 4. Build conversational chain
                        llm = get_gemini_llm(google_api_key=api_key, model_name=model_choice, temperature=temperature)
                        chain = build_conversational_rag_chain(llm=llm, retriever=retriever, window_k=5)

                        # Store in session state
                        st.session_state.chain = chain
                        st.session_state.indexed = True
                        st.session_state.doc_info = {
                            "filename": uploaded_file.name,
                            "pages": len(raw_docs),
                            "chunks": len(chunks)
                        }
                        st.session_state.messages = []  # Clear previous chat
                        st.success(f"Indexed **{uploaded_file.name}** successfully!")
                except Exception as e:
                    st.error(f"Error during indexing: {str(e)}")

    # Document Status Card
    if st.session_state.indexed and st.session_state.doc_info:
        st.markdown(
            f"""
            <div class="metric-card">
                <b>Active Document:</b> {st.session_state.doc_info['filename']}<br>
                <b>Pages:</b> {st.session_state.doc_info['pages']} | <b>Chunks:</b> {st.session_state.doc_info['chunks']}<br>
                <span style="color: #16A34A; font-weight: 600;">Status: Ready for Q&A</span>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.divider()

    # Session Reset Buttons
    col_reset_chat, col_reset_all = st.columns(2)
    with col_reset_chat:
        if st.button("🧹 Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    with col_reset_all:
        if st.button("🔄 Reset All", use_container_width=True):
            st.session_state.messages = []
            st.session_state.chain = None
            st.session_state.doc_info = None
            st.session_state.indexed = False
            st.rerun()


# Main Chat Area
st.markdown('<div class="main-header">📄 PDF Chatbot with LangChain & Google Gemini</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Conversational RAG pipeline with semantic search via FAISS and verifiable source citations.</div>', unsafe_allow_html=True)

st.markdown(
    """
    <div class="badge-container">
        <span class="tech-badge">Python 3.12</span>
        <span class="tech-badge">LangChain</span>
        <span class="tech-badge">Google Gemini 1.5</span>
        <span class="tech-badge">FAISS Vector Store</span>
        <span class="tech-badge">Streamlit</span>
        <span class="tech-badge">100% Free Tier</span>
    </div>
    """,
    unsafe_allow_html=True
)

# If no document is indexed, display welcome guides
if not st.session_state.indexed:
    st.info("👈 Upload your PDF in the left sidebar and click **'Process & Index PDF'** to start asking questions.")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            """
            #### 📊 Research & Papers
            Upload scientific papers, whitepapers, or literature reviews. Query methodology, results, and conclusions instantly.
            """
        )
    with col2:
        st.markdown(
            """
            #### 📑 Financial Reports
            Analyze 10-K filings, quarterly earnings, and annual reports. Extract key metrics, balance sheet stats, and guidance.
            """
        )
    with col3:
        st.markdown(
            """
            #### 📜 Contracts & Manuals
            Query dense legal contracts, compliance documents, or software technical manuals with page-level citations.
            """
        )
else:
    # Display message history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("citations"):
                with st.expander(f"📌 View Source Citations ({len(msg['citations'])} excerpts)"):
                    for idx, cit in enumerate(msg["citations"], start=1):
                        st.markdown(
                            f"""
                            <div class="source-box">
                                <b>Source #{idx}</b> — <i>{cit['source']}</i> (Page {cit['page']})<br>
                                <blockquote style="margin: 4px 0; color: #334155;">"{cit['preview']}"</blockquote>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

    # Chat Input
    if prompt := st.chat_input("Ask a question about the document..."):
        # Append and display user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response using RAG chain
        with st.chat_message("assistant"):
            with st.spinner("Searching document context and generating answer with Gemini..."):
                try:
                    result = st.session_state.chain.invoke({"question": prompt})
                    answer = result.get("answer", "No answer could be generated.")
                    source_docs = result.get("source_documents", [])
                    citations = format_citations(source_docs)

                    st.markdown(answer)

                    if citations:
                        with st.expander(f"📌 View Source Citations ({len(citations)} excerpts)"):
                            for idx, cit in enumerate(citations, start=1):
                                st.markdown(
                                    f"""
                                    <div class="source-box">
                                        <b>Source #{idx}</b> — <i>{cit['source']}</i> (Page {cit['page']})<br>
                                        <blockquote style="margin: 4px 0; color: #334155;">"{cit['preview']}"</blockquote>
                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )

                    # Append assistant message to session state
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "citations": citations
                    })

                except Exception as e:
                    error_msg = f"⚠️ An error occurred while generating the answer: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
