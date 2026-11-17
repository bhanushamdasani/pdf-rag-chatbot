# PDF Chatbot with Retrieval-Augmented Generation (RAG)

A conversational document question-answering system built with LangChain, FAISS, Google Gemini, and Streamlit. The application processes uploaded PDF documents, indexes their contents using dense vector representations, and provides grounded answers to user queries with verifiable page-level citations.

---

## Technical Overview

The system implements a standard Retrieval-Augmented Generation (RAG) architecture separated into two distinct pipelines:

1. **Document Ingestion & Vector Indexing Pipeline**:
   - Parses multi-page PDFs using `pdfplumber`, preserving page metadata for citation tracking.
   - Segments raw text into overlapping windows using `RecursiveCharacterTextSplitter` (default: 1,000 characters with 200 character overlap) to preserve contextual boundaries across chunk boundaries.
   - Embeds chunks using Google Gemini embeddings (`models/gemini-embedding-001`) and indexes them into an in-memory `FAISS` vector store using L2 distance.

2. **Conversational Query Pipeline**:
   - Maintains multi-turn conversational context using a sliding window buffer (`ConversationBufferWindowMemory`, $k=5$).
   - On subsequent turns, the chain rephrases follow-up questions into standalone queries using the conversation history before executing vector similarity search.
   - Retrieves the top-$k$ most relevant chunks (default: $k=4$) from FAISS.
   - Injects retrieved chunks into a strictly grounded system prompt.
   - Generates answers using Google Gemini (`gemini-3.8-flash`) with low temperature ($T=0.2$) to minimize hallucination.
   - Extracts and formats page numbers and snippet excerpts into structured citations returned alongside the response.

```
                              INGESTION PIPELINE
[PDF Upload] -> [pdfplumber Parser] -> [Recursive Chunking] -> [Gemini Embeddings] -> [FAISS Index]
                                                                                           |
                                QUERY PIPELINE                                            |
[User Query] -> [History Rephraser (k=5)] -> [Top-k Similarity Search] <-------------------+
                                                      |
                                                      v
                                        [Grounded Context + Prompt]
                                                      |
                                                      v
                                           [Gemini 1.5/3.8 Flash]
                                                      |
                                                      v
                                           [Answer + Page Citations]
```

---

## Key Design Decisions & Trade-offs

| Component | Choice | Rationale |
| :--- | :--- | :--- |
| **PDF Extraction** | `pdfplumber` | More robust handling of whitespace, layouts, and tables compared to `pypdf`, reducing noise during text chunking. |
| **Text Chunking** | `RecursiveCharacterTextSplitter` | Splits recursively on `["\n\n", "\n", " ", ""]` to keep related sentences and paragraphs together rather than cutting off arbitrarily mid-sentence. |
| **Vector Store** | `FAISS` (CPU) | Lightweight, self-contained, and runs entirely in memory without requiring external database infrastructure or cloud managed vector services. |
| **Conversational Memory** | `ConversationBufferWindowMemory` | Limits history to the last $k=5$ interactions to prevent prompt token bloat while retaining sufficient context for natural follow-ups. |
| **Grounding Strategy** | Constrained System Prompt | Instructs the model to strictly answer using the provided context and explicitly declare when the requested information is absent from the document. |
| **Offline Fallback** | `LocalTFIDFEmbeddings` & Extractive Engine | Provides an offline execution path that allows testing and demonstration without external API keys or network calls. |

---

## Project Structure

```
pdf-rag-chatbot/
├── .streamlit/
│   └── config.toml             # Streamlit server and theme configuration
├── src/
│   ├── __init__.py
│   ├── document_processor.py   # PDF text extraction and recursive chunking logic
│   ├── vector_store.py         # FAISS vector store and embedding wrappers
│   └── rag_chain.py            # ConversationalRetrievalChain, memory, and citation formatting
├── tests/
│   ├── __init__.py
│   └── test_rag_pipeline.py    # Unit tests covering chunking, indexing, memory, and citations
├── sample_docs/
│   └── Enterprise_AI_Report_2025.pdf  # 3-page sample PDF for testing and demonstration
├── scripts/
│   └── generate_sample_pdf.py  # Script used to generate reproducible sample PDFs
├── app.py                      # Streamlit UI with chat interface and citation drawer
├── pytest.ini                  # Pytest path and runner configuration
├── requirements.txt            # Dependency manifest
├── .env.example                # Template for environment variables
└── .gitignore                  # Git exclusion rules
```

---

## Getting Started

### Prerequisites
- Python 3.10+ (tested on Python 3.12)
- Git

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/bhanushamdasani/pdf-rag-chatbot.git
   cd pdf-rag-chatbot
   ```

2. Create and activate a virtual environment:
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate

   # macOS / Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

---

## Configuration

The application supports loading configuration via a local `.env` file, Streamlit Secrets, or manual input in the UI.

1. Copy the template:
   ```bash
   cp .env.example .env
   ```

2. Add your Google Gemini API key:
   ```env
   GOOGLE_API_KEY=your_gemini_api_key_here
   ```

*Note: The key is kept strictly in-memory and masked in the UI. The `.env` file is ignored by Git.*

---

## Running the Application

Launch the Streamlit web application:

```bash
streamlit run app.py
```

The application will be accessible at `http://localhost:8501`.

### Demonstration Workflow
1. In the sidebar under **Document Source**, click **"Quick Demo: Load Sample Report"** to load the pre-packaged 3-page benchmark report (`sample_docs/Enterprise_AI_Report_2025.pdf`).
2. Alternatively, upload any standard PDF file.
3. Once indexed, query the document using natural language in the chat input.
4. Expand the **Source Citations** drawer below assistant responses to verify the exact page numbers and excerpts used to answer.

---

## Running Automated Tests

The repository includes a `pytest` suite testing document chunking integrity, FAISS vector indexing, conversational memory state transitions, and citation generation.

Run tests using:

```bash
pytest
```

Test coverage includes:
- `test_chunk_documents`: Verifies document splitting boundaries and metadata preservation (`source`, `page`, `chunk_id`).
- `test_faiss_vector_store_and_retriever`: Verifies semantic vector search and top-$k$ retrieval.
- `test_conversational_rag_chain_with_memory`: Tests 2-turn query handling with question rephrasing and citation tracking.
- `test_offline_demo_pipeline`: Verifies end-to-end RAG execution using local embeddings without external API access.
- `test_empty_document_handling`: Ensures validation exceptions are raised for empty document payloads.

---

## Deployment (Streamlit Community Cloud)

To deploy with a public URL:

1. Push this repository to your GitHub account.
2. Navigate to [share.streamlit.io](https://share.streamlit.io) and link your GitHub account.
3. Select your repository (`pdf-rag-chatbot`), branch (`main`), and target file (`app.py`).
4. In **Advanced Settings -> Secrets**, add:
   ```toml
   GOOGLE_API_KEY = "your_gemini_api_key"
   ```
5. Click **Deploy**.

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
