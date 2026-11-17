"""PDF Chatbot using LangChain, FAISS, and Google Gemini.
Production-ready Streamlit application with conversational memory, source citations,
secure hidden key management, and 1-click live/offline demonstration mode.
"""

import os
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv

from src.document_processor import extract_text_from_pdf, chunk_documents
from src.vector_store import get_embedding_model, create_faiss_vector_store, get_retriever
from src.rag_chain import get_gemini_llm, build_conversational_rag_chain, format_citations

# Load environment variables from .env
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
        margin-bottom: 1.2rem;
    }
    .badge-container {
        display: flex;
        gap: 8px;
        margin-bottom: 1.2rem;
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
    .secure-badge {
        background-color: #ECFDF5;
        color: #059669;
        padding: 6px 12px;
        border-radius: 6px;
        font-size: 0.82rem;
        font-weight: 600;
        border: 1px solid #A7F3D0;
        margin-bottom: 10px;
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

if "demo_mode" not in st.session_state:
    st.session_state.demo_mode = False


# Helper function to index documents
def process_and_index_document(source, filename, api_key_val, demo_mode_val, chunk_sz, chunk_ovlp, top_k_val, model_name, temp):
    raw_docs = extract_text_from_pdf(source, filename=filename)
    if not raw_docs:
        raise ValueError("No extractable text found in the PDF document.")

    chunks = chunk_documents(raw_docs, chunk_size=chunk_sz, chunk_overlap=chunk_ovlp)
    embeddings = get_embedding_model(google_api_key=api_key_val, demo_mode=demo_mode_val)
    vector_store = create_faiss_vector_store(chunks, embeddings)
    retriever = get_retriever(vector_store, top_k=top_k_val)

    llm = get_gemini_llm(
        google_api_key=api_key_val,
        model_name=model_name,
        temperature=temp,
        demo_mode=demo_mode_val
    )
    chain = build_conversational_rag_chain(llm=llm, retriever=retriever, window_k=5)

    st.session_state.chain = chain
    st.session_state.indexed = True
    st.session_state.demo_mode = demo_mode_val
    st.session_state.doc_info = {
        "filename": filename,
        "pages": len(raw_docs),
        "chunks": len(chunks),
        "mode": "Offline Demo" if demo_mode_val else f"Live Gemini ({model_name})"
    }
    st.session_state.messages = []


# Sidebar Configuration
with st.sidebar:
    st.title("⚙️ Configuration")

    # SECURE API KEY HANDLING
    # Check secrets & environment without ever printing or echoing the key
    preconfigured_key = ""
    try:
        preconfigured_key = st.secrets.get("GOOGLE_API_KEY", "")
    except Exception:
        pass
    if not preconfigured_key:
        preconfigured_key = os.getenv("GOOGLE_API_KEY", "")

    effective_api_key = None
    manual_override = False

    st.subheader("🔑 API Key Security")
    if preconfigured_key:
        st.markdown(
            '<div class="secure-badge">🔒 API Key: Active & Protected (Hidden)</div>',
            unsafe_allow_html=True
        )
        manual_override = st.checkbox("Override with different key", value=False)
        if manual_override:
            effective_api_key = st.text_input(
                "Custom Gemini API Key",
                type="password",
                placeholder="Paste key here (will remain masked)"
            )
        else:
            effective_api_key = preconfigured_key
    else:
        effective_api_key = st.text_input(
            "Google Gemini API Key",
            type="password",
            placeholder="AIzaSy... (kept strictly masked)",
            help="Free key from Google AI Studio. Stored strictly in memory during your session."
        )
        st.info("💡 Get a free key with no credit card at [Google AI Studio](https://aistudio.google.com/app/apikey).")

    # Offline Demonstration Mode Switch
    offline_demo_toggle = st.toggle(
        "🧪 Offline Demo Mode (No API Key Required)",
        value=st.session_state.demo_mode or (not bool(effective_api_key)),
        help="Runs 100% offline using local TF-IDF semantic search and extractive summarization. Perfect for mock interviews and demonstrations."
    )

    st.divider()

    # Model Parameters
    st.subheader("🤖 Model & RAG Settings")
    model_choice = st.selectbox(
        "Gemini Model",
        options=["gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.0-flash"],
        index=0,
        disabled=offline_demo_toggle,
        help="gemini-1.5-flash is fast, accurate, and free-tier friendly."
    )

    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.2,
        step=0.05,
        disabled=offline_demo_toggle,
        help="Lower values produce more deterministic and grounded answers."
    )

    top_k = st.slider(
        "Top-k Retrievable Chunks",
        min_value=1,
        max_value=8,
        value=4,
        help="Number of relevant document chunks retrieved per query."
    )

    chunk_size = st.number_input("Chunk Size", min_value=300, max_value=3000, value=1000, step=100)
    chunk_overlap = st.number_input("Chunk Overlap", min_value=0, max_value=500, value=200, step=50)

    st.divider()

    # Document Upload & Quick Demo Section
    st.subheader("📑 Document Source")

    # 1-Click Demo Document Button
    sample_pdf_path = Path("sample_docs/Enterprise_AI_Report_2025.pdf")
    if sample_pdf_path.exists():
        if st.button("📄 Quick Demo: Load Sample Report", use_container_width=True):
            if not offline_demo_toggle and not effective_api_key:
                st.error("Please enter a Gemini API key or switch on 'Offline Demo Mode' above.")
            else:
                with st.spinner("Indexing sample report into FAISS vector store..."):
                    try:
                        process_and_index_document(
                            source=str(sample_pdf_path),
                            filename="Enterprise_AI_Report_2025.pdf",
                            api_key_val=effective_api_key,
                            demo_mode_val=offline_demo_toggle,
                            chunk_sz=chunk_size,
                            chunk_ovlp=chunk_overlap,
                            top_k_val=top_k,
                            model_name=model_choice,
                            temp=temperature
                        )
                        st.success("Loaded **Enterprise_AI_Report_2025.pdf**!")
                    except Exception as e:
                        st.error(f"Error loading sample: {str(e)}")

    uploaded_file = st.file_uploader(
        "Or Upload Your Own PDF",
        type=["pdf"],
        help="Upload any PDF to build the FAISS vector index."
    )

    if uploaded_file and st.button("🚀 Process & Index Uploaded PDF", use_container_width=True):
        if not offline_demo_toggle and not effective_api_key:
            st.error("Please enter your Gemini API key or switch on 'Offline Demo Mode' above.")
        else:
            with st.spinner("Processing document and generating vector index..."):
                try:
                    process_and_index_document(
                        source=uploaded_file,
                        filename=uploaded_file.name,
                        api_key_val=effective_api_key,
                        demo_mode_val=offline_demo_toggle,
                        chunk_sz=chunk_size,
                        chunk_ovlp=chunk_overlap,
                        top_k_val=top_k,
                        model_name=model_choice,
                        temp=temperature
                    )
                    st.success(f"Indexed **{uploaded_file.name}** successfully!")
                except Exception as e:
                    st.error(f"Error during indexing: {str(e)}")

    # Active Document Status Card
    if st.session_state.indexed and st.session_state.doc_info:
        st.markdown(
            f"""
            <div class="metric-card">
                <b>Document:</b> {st.session_state.doc_info['filename']}<br>
                <b>Pages:</b> {st.session_state.doc_info['pages']} | <b>Chunks:</b> {st.session_state.doc_info['chunks']}<br>
                <b>Pipeline:</b> {st.session_state.doc_info['mode']}<br>
                <span style="color: #16A34A; font-weight: 600;">Status: Ready for Q&A</span>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.divider()

    # Session Management
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
st.markdown('<div class="sub-header">Conversational RAG pipeline with semantic vector search and verifiable source page citations.</div>', unsafe_allow_html=True)

st.markdown(
    """
    <div class="badge-container">
        <span class="tech-badge">Python 3.12</span>
        <span class="tech-badge">LangChain</span>
        <span class="tech-badge">Google Gemini 1.5</span>
        <span class="tech-badge">FAISS Vector Store</span>
        <span class="tech-badge">Streamlit</span>
        <span class="tech-badge">Zero Cost ($0)</span>
    </div>
    """,
    unsafe_allow_html=True
)

# Helper function to submit query
def submit_question(user_question: str):
    st.session_state.messages.append({"role": "user", "content": user_question})
    try:
        result = st.session_state.chain.invoke({"question": user_question})
        answer = result.get("answer", "No answer could be generated.")
        source_docs = result.get("source_documents", [])
        citations = format_citations(source_docs)

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "citations": citations
        })
    except Exception as e:
        error_msg = f"⚠️ An error occurred while generating the answer: {str(e)}"
        st.session_state.messages.append({"role": "assistant", "content": error_msg})


# If no document is indexed, display quick demonstration options
if not st.session_state.indexed:
    st.info("👈 Click **'📄 Quick Demo: Load Sample Report'** in the left sidebar or upload your own PDF to begin.")

    st.subheader("💡 What you can demonstrate with this project:")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            """
            #### 1. Semantic Vector Search
            Dense similarity retrieval powered by FAISS with millisecond response latency and configurable chunk overlap.
            """
        )
    with col2:
        st.markdown(
            """
            #### 2. Multi-Turn Memory
            Conversation sliding window buffer maintaining dialogue context across follow-up queries.
            """
        )
    with col3:
        st.markdown(
            """
            #### 3. Verifiable Citations
            Anti-hallucination grounded pipeline providing exact page-level source references for auditability.
            """
        )

else:
    # Display Sample Prompt Pills if chatting with sample report
    if st.session_state.doc_info and "Enterprise_AI_Report" in st.session_state.doc_info["filename"]:
        st.caption("✨ **Quick Demonstration Questions (Click to ask):**")
        pcol1, pcol2, pcol3, pcol4 = st.columns(4)
        with pcol1:
            if st.button("📈 Adoption Rate in 2025?", use_container_width=True):
                submit_question("What was the enterprise adoption rate of RAG in 2025?")
                st.rerun()
        with pcol2:
            if st.button("💰 Project Titan Funding?", use_container_width=True):
                submit_question("How much research grant funding was allocated to Project Titan?")
                st.rerun()
        with pcol3:
            if st.button("⚡ FAISS Vector Latency?", use_container_width=True):
                submit_question("What was the average retrieval latency achieved by FAISS?")
                st.rerun()
        with pcol4:
            if st.button("🛡️ Security Standards?", use_container_width=True):
                submit_question("What compliance and security standards does the RAG system satisfy?")
                st.rerun()

    # Display Message History
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("citations"):
                with st.expander(f"📌 View Source Citations ({len(msg['citations'])} page references)"):
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

    # Chat Input Box
    if prompt := st.chat_input("Ask a question about your document..."):
        submit_question(prompt)
        st.rerun()
