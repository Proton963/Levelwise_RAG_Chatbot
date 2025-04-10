import os
import tempfile
import traceback
import concurrent.futures # Import for parallel processing
from typing import List, Any, Optional, Tuple, Iterable
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
# Import document loaders
from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader,
    CSVLoader, UnstructuredExcelLoader
)

# --- File Loading Helper (Same as before) ---
def _load_documents_from_file(file_path: str, file_extension: str, splitter: RecursiveCharacterTextSplitter) -> List[Document]:
    """Internal function to load and split documents based on file type."""
    documents = []
    loader: Any = None
    print(f"DEBUG (embedding_manager): Loading documents from path: {file_path} (type: {file_extension})")
    try:
        if file_extension == ".pdf":
            loader = PyPDFLoader(file_path)
            documents = loader.load_and_split(text_splitter=splitter)
        elif file_extension == ".txt":
            loader = TextLoader(file_path, encoding='utf-8')
            documents = loader.load_and_split(text_splitter=splitter)
        elif file_extension == ".docx":
             loader = UnstructuredWordDocumentLoader(file_path)
             documents = loader.load_and_split(text_splitter=splitter)
        elif file_extension == ".csv":
             loader = CSVLoader(file_path)
             documents = loader.load() # May need custom splitting logic for rows
        elif file_extension == ".xlsx":
              loader = UnstructuredExcelLoader(file_path, mode="elements")
              documents = loader.load_and_split(text_splitter=splitter)
        else:
             print(f"ERROR: Unsupported file type '{file_extension}' in loader.")
             raise ValueError(f"Unsupported file type: {file_extension}")
        print(f"DEBUG (embedding_manager): Loaded {len(documents)} document chunks.")
        return documents
    except Exception as e:
         print(f"ERROR (embedding_manager): Failed during document loading for {file_path}: {e}")
         traceback.print_exc()
         return []

# --- Batching Helpers ---
def _split_texts_into_batches(texts: List[str], batch_size: int) -> Iterable[List[str]]:
    """Splits a list of texts into batches."""
    if not texts:
        return
    for i in range(0, len(texts), batch_size):
        yield texts[i:i + batch_size]

def _embed_text_batch(text_batch: List[str], embedding_model: Any) -> List[List[float]]:
    """Embeds a single batch of texts, handling potential errors."""
    try:
        # Directly call embed_documents on the batch of texts
        embeddings = embedding_model.embed_documents(text_batch)
        print(f"DEBUG (embedding_manager): Embedded batch of {len(text_batch)} texts.")
        # Check if the number of embeddings matches the number of texts
        if len(embeddings) == len(text_batch):
            return embeddings
        else:
            print(f"WARN (embedding_manager): Embedding count mismatch for batch. Got {len(embeddings)}, expected {len(text_batch)}. Returning partial/empty.")
            # Handle mismatch - return empties for safety, or try to align based on available data
            return [[] for _ in text_batch] # Safest default: return list of empty lists on mismatch
    except Exception as e:
        print(f"ERROR (embedding_manager): Embedding batch failed: {e}. Returning empty lists.")
        # Return empty lists matching the batch size on error
        return [[] for _ in text_batch]

# --- Main function with Batch Processing ---
def process_uploaded_file_to_faiss(
    uploaded_file: Any, # Streamlit UploadedFile object
    embedding_model: Any, # Pre-initialized embedding model from app.py
    splitter: RecursiveCharacterTextSplitter, # Pre-initialized splitter from app.py
    batch_size: int = 32, # How many texts to send in one API call (adjust based on performance/API limits)
    max_workers: Optional[int] = 4 # Number of parallel threads for API calls (adjust based on CPU/network)
) -> Optional[FAISS]:
    """
    Reads file, loads/chunks, embeds content IN BATCHES (potentially parallel),
    and returns a FAISS index, or None on failure.
    """
    faiss_index = None
    tmp_file_path = None
    file_name = getattr(uploaded_file, 'name', 'unknown_file')
    print(f"DEBUG (embedding_manager): Starting BATCH processing for: {file_name}")

    try:
        # 1. Save uploaded file temporarily
        file_extension = os.path.splitext(file_name)[1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        print(f"DEBUG (embedding_manager): Temp file: {tmp_file_path}")

        # 2. Load and Split Documents
        # Assuming documents are reasonably chunked by the loader+splitter
        documents = _load_documents_from_file(tmp_file_path, file_extension, splitter)

        if documents:
            # 3. Prepare list of all texts to be embedded
            all_texts = [doc.page_content for doc in documents if doc.page_content and isinstance(doc.page_content, str)]
            if not all_texts:
                 print("ERROR (embedding_manager): No valid text content found after loading.")
                 raise ValueError("No text content extracted from file.")

            print(f"DEBUG (embedding_manager): Starting embedding for {len(all_texts)} chunks in batches of {batch_size}...")
            all_embeddings = []
            text_batches = list(_split_texts_into_batches(all_texts, batch_size))

            # Use ThreadPoolExecutor for potentially parallel embedding requests
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Map embedding function to batches
                results = list(executor.map(lambda batch: _embed_text_batch(batch, embedding_model), text_batches))

            # Flatten the list of lists of embeddings
            failed_batches = 0
            for i, batch_embeddings in enumerate(results):
                # Check for errors indicated by empty lists (or lists not matching batch size)
                original_batch_size = len(text_batches[i])
                if not batch_embeddings or len(batch_embeddings) != original_batch_size or not all(batch_embeddings):
                    print(f"WARN (embedding_manager): Batch {i+1} may have failed embedding. Expected {original_batch_size}, got {len(batch_embeddings)} valid embeddings.")
                    failed_batches += 1
                    # Pad with None or handle appropriately if you need perfect alignment
                    # For now, we'll just extend with valid ones, but this breaks alignment if some fail mid-batch
                    all_embeddings.extend([emb for emb in batch_embeddings if emb])
                else:
                    all_embeddings.extend(batch_embeddings)

            print(f"DEBUG (embedding_manager): Total embeddings collected: {len(all_embeddings)}. Expected approx: {len(all_texts)}. Failed batches: {failed_batches}")

            # 4. Create FAISS Index
            # Important: If any batch failed, len(all_embeddings) might not equal len(all_texts)
            # We should only index texts that were successfully embedded.
            # Rebuild the text list based on successful embeddings if needed, or filter pairs.
            # Simple approach: Ensure counts match before proceeding.
            if all_texts and all_embeddings and len(all_texts) == len(all_embeddings):
                try:
                    print(f"DEBUG (embedding_manager): Creating FAISS index from {len(all_texts)} text/embedding pairs...")
                    text_embedding_pairs = list(zip(all_texts, all_embeddings))
                    faiss_index = FAISS.from_embeddings(text_embedding_pairs, embedding_model)
                    print("DEBUG (embedding_manager): FAISS index created successfully.")
                except Exception as e:
                    print(f"ERROR (embedding_manager): creating FAISS index: {e}")
                    traceback.print_exc()
            else:
                # Log mismatch for debugging
                print(f"ERROR (embedding_manager): Mismatch between texts ({len(all_texts)}) and collected embeddings ({len(all_embeddings)}). Cannot create FAISS index reliably.")

        else:
             print(f"WARN (embedding_manager): No documents loaded from file {file_name}.")

    except Exception as e:
         print(f"ERROR (embedding_manager): processing file '{file_name}': {e}")
         traceback.print_exc()
    finally:
         # 5. Clean up temporary file
         if tmp_file_path and os.path.exists(tmp_file_path):
             try:
                 os.remove(tmp_file_path)
                 print(f"DEBUG (embedding_manager): Removed temp file: {tmp_file_path}")
             except Exception as e_rem:
                 print(f"ERROR (embedding_manager): removing temp file {tmp_file_path}: {e_rem}")

    return faiss_index