"""Document processing module: extracts text from PDFs and performs intelligent chunking."""

import io
from typing import List, Union, BinaryIO
import pdfplumber
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def extract_text_from_pdf(
    file_source: Union[str, BinaryIO, io.BytesIO],
    filename: str = "uploaded_doc.pdf"
) -> List[Document]:
    """
    Extracts text page-by-page from a PDF file using pdfplumber.
    Preserves page numbers in document metadata for citation tracking.

    Args:
        file_source: Path to file or binary file-like object (e.g. from st.file_uploader)
        filename: Optional name of the file for metadata tracking

    Returns:
        List of LangChain Document objects with page content and metadata.
    """
    documents: List[Document] = []

    with pdfplumber.open(file_source) as pdf:
        for page_index, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            # Strip extra whitespace but maintain paragraph structure
            cleaned_text = text.strip()
            if cleaned_text:
                doc = Document(
                    page_content=cleaned_text,
                    metadata={
                        "source": filename,
                        "page": page_index,
                        "total_pages": len(pdf.pages),
                    }
                )
                documents.append(doc)

    return documents


def chunk_documents(
    documents: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> List[Document]:
    """
    Splits documents into overlapping chunks using RecursiveCharacterTextSplitter.
    Preserves original page and source metadata in every chunk.

    Args:
        documents: List of input Document objects
        chunk_size: Maximum characters per chunk
        chunk_overlap: Number of characters to overlap between adjacent chunks

    Returns:
        List of chunked Document objects.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)
    
    # Assign chunk ids for better traceability
    for idx, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = idx
        
    return chunks

