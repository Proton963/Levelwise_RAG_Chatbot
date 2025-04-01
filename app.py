import os
import datetime
import streamlit as st
from pymongo import MongoClient
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langchain.retrievers import EnsembleRetriever
from chat_manager.chat_manager import build_chat_history, get_retrieval_chain
from ui.ui import (
    login_page,
    display_user_info,
    setup_sidebar,
    setup_sidebar_progress_bar,
    show_vector_store_ready,
    update_sidebar_status,
    display_project_title_and_logo,
    display_chat_history,
    display_chat_message
)
from access_control.access_control import store_uploaded_file
from pydantic import SecretStr

# -----------------------------------------------------------------------------
# 1. Check if user is logged in; if not, show login page with logo.
# -----------------------------------------------------------------------------
logo_path = "D:/LevelWise_RAG/assets/LevelWiseRAG.png"  # Adjust if needed

if "logged_in" not in st.session_state or not st.session_state.logged_in:
    login_page(logo_path)  # This shows the top-left corner logo and name
    st.stop()

# -----------------------------------------------------------------------------
# 2. Load Environment Variables
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
# 3. Connect to MongoDB
# -----------------------------------------------------------------------------
client = MongoClient(mongo_uri)
db = client[mongo_db_name]

# -----------------------------------------------------------------------------
# 4. Retrieve Current User and Display in Sidebar
# -----------------------------------------------------------------------------
current_user = st.session_state.current_user
display_user_info()

# -----------------------------------------------------------------------------
# 5. Initialize LLM, Embedding Model, and Text Splitter
# -----------------------------------------------------------------------------
llm = ChatGroq(api_key=SecretStr(groq_api_key), model="gemma2-9b-it", stop_sequences=[])
embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)

# -----------------------------------------------------------------------------
# 6. Build Text Blocks for Each Collection Document
# -----------------------------------------------------------------------------
def build_department_text(doc):
    dept_name = doc.get("name", "N/A")
    hod = doc.get("HOD", {})
    hod_name = hod.get("name", "N/A")
    hod_email = hod.get("email", "N/A")
    return f"Department: {dept_name}\nHOD Name: {hod_name}\nHOD Email: {hod_email}"

def build_professor_text(doc):
    name = doc.get("name", "N/A")
    dept_id = doc.get("department_id", "N/A")
    email = doc.get("email", "N/A")
    return f"Professor Name: {name}\nDepartment ID: {dept_id}\nEmail: {email}"

def build_student_text(doc):
    name = doc.get("name", "N/A")
    dept_id = doc.get("department_id", "N/A")
    email = doc.get("email", "N/A")
    return f"Student Name: {name}\nDepartment ID: {dept_id}\nEmail: {email}"

build_text_map = {
    "departments": build_department_text,
    "professors": build_professor_text,
    "students": build_student_text
}
collections = ["departments", "professors", "students"]

# -----------------------------------------------------------------------------
# 7. Create and Load FAISS Indexes for MongoDB Collections
# -----------------------------------------------------------------------------
@st.cache_resource
def create_and_load_indexes():
    vector_stores = {}
    print("DEBUG: Starting index creation/loading process...")
    for coll_name in collections:
        print(f"DEBUG: Processing collection '{coll_name}'...")
        index_dir = f"mongodb_faiss_index_{coll_name}"
        index_path = os.path.join(index_dir, "index.faiss")
        text_blocks = []
        metadatas = []
        build_func = build_text_map.get(coll_name)
        if not os.path.exists(index_path):
            print(f"DEBUG: Index for '{coll_name}' not found. Creating index...")
            collection = db[coll_name]
            documents = list(collection.find())
            if build_func:
                for doc in documents:
                    text = build_func(doc)
                    if text.strip():
                        text_blocks.append(text)
                        if coll_name in ["students", "professors"]:
                            dept_val = doc.get("department_id", "").strip()
                            metadatas.append({"department": dept_val})
            if text_blocks:
                if coll_name in ["students", "professors"]:
                    vector_store = FAISS.from_texts(text_blocks, embedding_model, metadatas=metadatas)
                else:
                    combined_text = "\n".join(text_blocks)
                    chunks = splitter.split_text(combined_text)
                    vector_store = FAISS.from_texts(chunks, embedding_model)
                vector_store.save_local(index_dir)
                print(f"DEBUG: Index for '{coll_name}' created and saved to '{index_dir}'.")
            else:
                print(f"DEBUG: No text data for '{coll_name}'. Skipping index creation.")
        try:
            vector_stores[coll_name] = FAISS.load_local(
                index_dir,
                embedding_model,
                allow_dangerous_deserialization=True
            )
            print(f"DEBUG: Loaded index for '{coll_name}'.")
        except Exception as e:
            print(f"DEBUG: Failed to load index for '{coll_name}': {e}")
            vector_stores[coll_name] = None
    print("DEBUG: Finished index creation/loading process.")
    return vector_stores

all_mongodb_indexes = create_and_load_indexes()

# -----------------------------------------------------------------------------
# 8. Filter Data Sources Based on User Role
# -----------------------------------------------------------------------------
def select_indexes_for_user(all_indexes, user):
    role = user.get("role")
    if role == "HOD":
        return all_indexes
    elif role == "Professor":
        return {k: v for k, v in all_indexes.items() if k in ["professors", "students"]}
    elif role == "Student":
        # Let students see both "students" and "professors" from their department
        return {k: v for k, v in all_indexes.items() if k in ["students", "professors"]}
    else:
        return {}

filtered_indexes = select_indexes_for_user(all_mongodb_indexes, current_user)
print("DEBUG: Filtered indexes based on user role:", filtered_indexes)

# -----------------------------------------------------------------------------
# 9. Build Retrievers with Department Filter
# -----------------------------------------------------------------------------
retrievers = []
for key, vs in filtered_indexes.items():
    if vs is not None:
        if key in ["students", "professors"] and current_user.get("role") in ["Student", "Professor"]:
            retriever = vs.as_retriever(search_kwargs={"filter": {"department": current_user.get("department")}})
            print(f"DEBUG: Added '{key}' retriever with department filter: {current_user.get('department')}")
        else:
            retriever = vs.as_retriever()
        retrievers.append(retriever)

ensemble_retriever = None
if not retrievers:
    st.error("No valid retrievers loaded. Please check your indexes.")
else:
    ensemble_retriever = EnsembleRetriever(
        retrievers=retrievers,
        weights=[1.0 for _ in retrievers]
    )

# -----------------------------------------------------------------------------
# 10. (Optional) Display Title/Logo on Main Page
# -----------------------------------------------------------------------------
# If you also want the same top-left logo on the main page, uncomment:
display_project_title_and_logo(logo_path)

# -----------------------------------------------------------------------------
# 11. Sidebar: File Upload
# -----------------------------------------------------------------------------
uploaded_file = st.sidebar.file_uploader("Upload a document (PDF, TXT, Excel, DOCS, CSV)", type=["pdf", "txt", "xlsx", "doc", "csv"])
if uploaded_file:
    file_info = store_uploaded_file(uploaded_file, current_user)
    st.sidebar.success(f"File '{file_info['filename']}' uploaded successfully!")
    print(f"DEBUG: File '{file_info['filename']}' uploaded by {current_user.get('username')}.")

# -----------------------------------------------------------------------------
# 12. Initialize Session State for Chat Messages
# -----------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "How can I help you?"}]

# -----------------------------------------------------------------------------
# 13. Main Chat Interface
# -----------------------------------------------------------------------------
display_chat_history(st.session_state.messages)
user_input = st.chat_input("Ask your question:")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    display_chat_message("user", user_input)
    chat_history = build_chat_history(st.session_state.messages)
    if ensemble_retriever is not None:
        rag_chain = get_retrieval_chain(llm, ensemble_retriever, current_user=current_user)
        response = rag_chain.invoke({'input': user_input, 'chat_history': chat_history})
        answer = response['answer']
        st.session_state.messages.append({"role": "assistant", "content": answer})
        display_chat_message("assistant", answer)
    else:
        st.error("No valid retriever available for query processing.")
