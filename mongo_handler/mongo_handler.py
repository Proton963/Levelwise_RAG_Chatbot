import os
from pymongo import MongoClient
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

# Load environment variables (for GOOGLE_API_KEY, MONGO_URI, and MONGO_DB_NAME)
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    raise ValueError("GOOGLE_API_KEY not found in environment variables.")

mongo_uri = os.getenv("MONGO_URI")
if not mongo_uri:
    raise ValueError("MONGO_URI not found in environment variables.")

mongo_db_name = os.getenv("MONGO_DB_NAME", "collegeDB")

# Connect to MongoDB
client = MongoClient(mongo_uri)
db = client[mongo_db_name]

# Initialize text splitter and embedding model
splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Retrieve all collection names from the database
collection_names = db.list_collection_names()

for coll_name in collection_names:
    collection = db[coll_name]
    documents = list(collection.find())
    # Extract text from each document (assumes documents have a "text" field)
    texts = [doc["text"] for doc in documents if "text" in doc]
    
    if texts:
        # Combine texts and split into chunks
        chunks = splitter.split_text("\n".join(texts))
        # Create FAISS vector store for this collection
        vector_store = FAISS.from_texts(chunks, embeddings)
        # Save the index with a unique filename per collection
        index_filename = f"mongodb_faiss_index_{coll_name}"
        vector_store.save_local(index_filename)
        print(f"Index for collection '{coll_name}' created and saved to '{index_filename}'.")
    else:
        print(f"No text data found in collection '{coll_name}'. Skipping index creation.")
