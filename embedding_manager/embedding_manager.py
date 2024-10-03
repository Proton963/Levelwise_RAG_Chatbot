from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

def create_embeddings(documents):
    # Split the documents into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)
    
    # Create embeddings for the document chunks
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    # Create a FAISS vector store from the embeddings
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store
