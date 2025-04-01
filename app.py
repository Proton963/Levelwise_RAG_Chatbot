import os
import streamlit as st
from pymongo import MongoClient
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langchain.retrievers import EnsembleRetriever
from document_handler.document_handler import load_documents
from embedding_manager.embedding_manager import create_embeddings_in_batches
from chat_manager.chat_manager import build_chat_history, get_retrieval_chain
from ui.ui import (
    setup_sidebar,
    display_project_title_and_logo,
    setup_sidebar_progress_bar,
    show_vector_store_ready,
    update_sidebar_status,
    display_chat_history,
    display_chat_message
)
from pydantic import SecretStr

# -----------------------------------------------------------------------------
# 1. Load Environment Variables and Validate
# -----------------------------------------------------------------------------
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
mongo_uri = os.getenv("MONGO_URI")
mongo_db_name = os.getenv("MONGO_DB_NAME", "collegeDB")

if not groq_api_key:
    st.error("GROQ_API_KEY environment variable not found. Please set it.")
    st.stop()
if not mongo_uri:
    st.error("MONGO_URI environment variable not found. Please set it.")
    st.stop()

# -----------------------------------------------------------------------------
# 2. Connect to MongoDB
# -----------------------------------------------------------------------------
client = MongoClient(mongo_uri)
db = client[mongo_db_name]

# -----------------------------------------------------------------------------
# 3. Initialize LLM, Embedding Model, and Text Splitter
# -----------------------------------------------------------------------------
llm = ChatGroq(api_key=SecretStr(groq_api_key), model="gemma2-9b-it", stop_sequences=[])
embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)

# -----------------------------------------------------------------------------
# 4. Build Text Blocks for Each Collection Document
# -----------------------------------------------------------------------------
def build_department_text(doc):
    """
    Build text for a department document.
    Schema: {_id, name, HOD: {name, email}}
    """
    dept_name = doc.get("name", "N/A")
    hod = doc.get("HOD", {})
    hod_name = hod.get("name", "N/A")
    hod_email = hod.get("email", "N/A")
    return f"Department: {dept_name}\nHOD Name: {hod_name}\nHOD Email: {hod_email}"

def build_professor_text(doc):
    """
    Build text for a professor document.
    Schema: {_id, name, department_id, email}
    """
    name = doc.get("name", "N/A")
    dept_id = doc.get("department_id", "N/A")
    email = doc.get("email", "N/A")
    return f"Professor Name: {name}\nDepartment ID: {dept_id}\nEmail: {email}"

def build_student_text(doc):
    """
    Build text for a student document.
    Schema: {_id, name, department_id, email}
    """
    name = doc.get("name", "N/A")
    dept_id = doc.get("department_id", "N/A")
    email = doc.get("email", "N/A")
    return f"Student Name: {name}\nDepartment ID: {dept_id}\nEmail: {email}"

# Mapping of collection names to their text builder functions
build_text_map = {
    "departments": build_department_text,
    "professors": build_professor_text,
    "students": build_student_text
}

collections = ["departments", "professors", "students"]

# -----------------------------------------------------------------------------
# 5. Create and Load FAISS Indexes for MongoDB Collections
# -----------------------------------------------------------------------------
@st.cache_resource
def create_and_load_indexes():
    """Create (if missing) and load FAISS indexes for each MongoDB collection with a single-line status."""
    vector_stores = {}
    
    # A single placeholder and progress bar in the sidebar
    status_placeholder = st.sidebar.empty()
    progress_bar = st.sidebar.progress(0)
    
    # List of your collections (already defined in the code)
    collections = ["departments", "professors", "students"]
    total = len(collections)
    
    status_placeholder.info("Starting index creation/loading...")
    
    for i, coll_name in enumerate(collections, start=1):
        # Update progress
        progress_percent = int((i - 1) / total * 100)
        progress_bar.progress(progress_percent)
        
        # Overwrite the same placeholder instead of creating new messages
        status_placeholder.info(f"Processing '{coll_name}'...")
        
        index_dir = f"mongodb_faiss_index_{coll_name}"
        index_path = os.path.join(index_dir, "index.faiss")
        
        # If index doesn't exist, create it
        if not os.path.exists(index_path):
            # Build text blocks from your documents
            collection = db[coll_name]
            documents = list(collection.find())
            
            # Use the appropriate build_text function (assuming you have them)
            build_func = build_text_map.get(coll_name)
            text_blocks = []
            if build_func is not None:
                for doc in documents:
                    text_block = build_func(doc)
                    if text_block.strip():
                        text_blocks.append(text_block)
            else:
                status_placeholder.warning(f"No text builder function found for '{coll_name}'. Skipping.")
            
            if text_blocks:
                combined_text = "\n".join(text_blocks)
                chunks = splitter.split_text(combined_text)
                vector_store = FAISS.from_texts(chunks, embedding_model)
                vector_store.save_local(index_dir)
            # If there's no data, we skip creating the index
            
        # Attempt to load the index (if it exists now)
        try:
            vector_stores[coll_name] = FAISS.load_local(
                index_dir,
                embedding_model,
                allow_dangerous_deserialization=True
            )
        except Exception as e:
            # If loading fails, set None so we skip it later
            vector_stores[coll_name] = None
    
    # Finish progress and update final status
    progress_bar.progress(100)
    status_placeholder.success("Index creation/loading complete.")
    
    return vector_stores

mongodb_vector_stores = create_and_load_indexes()

# Build retrievers from the loaded vector stores
retrievers = []
for vs in mongodb_vector_stores.values():
    if vs is not None:
        retrievers.append(vs.as_retriever())

# -----------------------------------------------------------------------------
# 6. Add PDF Vector Store Retriever if Available
# -----------------------------------------------------------------------------
if st.session_state.get("docs_loaded", False):
    pdf_retriever = st.session_state.vector_store.as_retriever()
    retrievers.append(pdf_retriever)

# -----------------------------------------------------------------------------
# 7. Create Ensemble Retriever from Available Retrievers
# -----------------------------------------------------------------------------
ensemble_retriever = None
if not retrievers:
    st.error("No valid retrievers loaded. Please check your indexes or upload PDF documents.")
else:
    ensemble_retriever = EnsembleRetriever(
        retrievers=retrievers,
        weights=[1.0 for _ in retrievers]  # Equal weights for each retriever
    )

# -----------------------------------------------------------------------------
# 8. UI: Display Project Title and Logo on Main Screen
# -----------------------------------------------------------------------------
logo_path = "D:/LevelWise_RAG/assets/LevelWiseRAG.png"  # Adjust this path as needed
display_project_title_and_logo(logo_path)

# -----------------------------------------------------------------------------
# 9. Sidebar: PDF Upload Controls
# -----------------------------------------------------------------------------
uploaded_files, create_vector_store_button = setup_sidebar()
if uploaded_files and create_vector_store_button:
    pdf_progress_bar, _ = setup_sidebar_progress_bar()  # re-use a progress bar for PDFs
    documents = load_documents(uploaded_files, upload_dir="uploads")
    batch_size = 5
    vector_store = create_embeddings_in_batches(documents, batch_size=batch_size)
    st.session_state.vector_store = vector_store
    st.session_state.docs_loaded = True
    pdf_progress_bar.progress(100)
    show_vector_store_ready()

# -----------------------------------------------------------------------------
# 10. Initialize Session State for Chat Messages
# -----------------------------------------------------------------------------
if "docs_loaded" not in st.session_state:
    st.session_state.docs_loaded = False
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "How can I help you?"}]

# -----------------------------------------------------------------------------
# 11. Main Chat Interface: Display Chat History and Process Input
# -----------------------------------------------------------------------------
display_chat_history(st.session_state.messages)
user_input = st.chat_input("Ask your question:")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    display_chat_message("user", user_input)
    chat_history = build_chat_history(st.session_state.messages)
    
    if ensemble_retriever is not None:
        rag_chain = get_retrieval_chain(llm, ensemble_retriever)
        response = rag_chain.invoke({'input': user_input, 'chat_history': chat_history})
        answer = response['answer']
        st.session_state.messages.append({"role": "assistant", "content": answer})
        display_chat_message("assistant", answer)
    else:
        st.error("No valid retriever available for query processing.")
