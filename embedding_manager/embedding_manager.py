from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import concurrent.futures

def split_into_batches(documents, batch_size):
    """Splits the documents into batches of a specified size."""
    for i in range(0, len(documents), batch_size):
        yield documents[i:i + batch_size]

def process_batch(batch, embedding_model):
    """Processes a batch of documents to create embeddings."""
    # Split documents into smaller chunks for embedding
    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
    chunks = splitter.split_documents(batch)
    
    # Extract text content from each chunk for embedding
    chunk_texts = [chunk.page_content for chunk in chunks]
    
    # Generate embeddings for each chunk using GoogleGenerativeAIEmbeddings
    embeddings = embedding_model.embed_documents(chunk_texts)  # Embed all texts in the batch
    return chunk_texts, embeddings  # Return texts and their embeddings

def create_embeddings_in_batches(documents, batch_size=5):
    """Creates embeddings for documents in parallel using batch processing."""
    # Initialize the embedding model
    embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    all_texts = []
    all_embeddings = []

    # Split documents into batches
    batches = list(split_into_batches(documents, batch_size))
    
    # Use ThreadPoolExecutor for parallel processing of batches
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_batch, batch, embedding_model) for batch in batches]
        
        # Collect results as they complete
        for future in concurrent.futures.as_completed(futures):
            texts, embeddings = future.result()
            all_texts.extend(texts)
            all_embeddings.extend(embeddings)

    # Create a single FAISS vector store with the combined embeddings and texts
    final_vector_store = FAISS.from_texts(all_texts, embedding_model)
    return final_vector_store
