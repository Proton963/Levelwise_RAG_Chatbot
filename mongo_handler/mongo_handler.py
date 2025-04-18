# # mongo_handler.py
import os
import traceback
import streamlit as st
from pymongo import MongoClient
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings # Keep for type hint, or use typing.Any
from langchain.text_splitter import RecursiveCharacterTextSplitter # Keep for type hint, or use typing.Any
from typing import List, Dict, Any, Optional

# --- Helper functions to format MongoDB documents into text ---
def build_department_text(doc: Dict[str, Any]) -> str:
    dept_name = doc.get("name", "N/A")
    hod = doc.get("HOD", {})
    hod_name = hod.get("name", "N/A")
    hod_email = hod.get("email", "N/A")
    return f"Department: {dept_name}\nHOD Name: {hod_name}\nHOD Email: {hod_email}"

def build_professor_text(doc: Dict[str, Any]) -> str:
    name = doc.get("name", "N/A")
    dept_id = doc.get("department_id", "N/A")
    email = doc.get("email", "N/A")
    return f"[Professor] Name: {name}\nDepartment ID: {dept_id}\nEmail: {email}"

def build_student_text(doc: Dict[str, Any]) -> str:
    name = doc.get("name", "N/A")
    dept_id = doc.get("department_id", "N/A")
    email = doc.get("email", "N/A")
    return f"[Student] Name: {name}\nDepartment ID: {dept_id}\nEmail: {email}"

# Map collection names to their respective build functions
build_text_map = {
    "departments": build_department_text,
    "professors": build_professor_text,
    "students": build_student_text
}

# --- Main function for index creation/loading ---
@st.cache_resource # Keep caching here for efficiency
def load_or_create_faiss_indexes(
    mongo_uri: str,
    mongo_db_name: str,
    collections_to_index: List[str],
    _embedding_model: Any, # Use 'Any' or specific embedding type
    _splitter: Optional[RecursiveCharacterTextSplitter] = None # Make splitter optional if not always used
) -> Dict[str, Optional[FAISS]]:
    """
    Connects to MongoDB, fetches data, creates/loads FAISS indexes for specified collections.

    Args:
        mongo_uri: MongoDB connection string.
        mongo_db_name: Name of the MongoDB database.
        collections_to_index: List of collection names to index (e.g., ["students", "professors"]).
        embedding_model: Initialized LangChain embedding model instance.
        splitter: Optional Initialized LangChain text splitter instance (used for 'departments').

    Returns:
        A dictionary mapping collection names to loaded FAISS vector store instances (or None if failed).
    """
    embedding_model = _embedding_model
    splitter = _splitter
    vector_stores = {}
    try:
        print(f"DEBUG: Connecting to MongoDB at {mongo_uri} / DB: {mongo_db_name}")
        client = MongoClient(mongo_uri)
        db = client[mongo_db_name]
        print("DEBUG: MongoDB connection successful.")
    except Exception as e:
        st.error(f"Failed to connect to MongoDB: {e}")
        print(f"ERROR: Failed to connect to MongoDB: {e}")
        return vector_stores # Return empty dict on connection failure

    print("DEBUG: Starting index creation/loading process...")
    for coll_name in collections_to_index:
        print(f"DEBUG: Processing collection '{coll_name}'...")
        index_dir = f"mongodb_faiss_index_{coll_name}" # Local directory to save/load index
        index_path = os.path.join(index_dir, "index.faiss") # Standard FAISS file name
        text_blocks = []
        metadatas = []
        build_func = build_text_map.get(coll_name)

        if not os.path.exists(index_path):
            print(f"DEBUG: Index for '{coll_name}' not found at '{index_dir}'. Creating index...")
            try:
                collection = db[coll_name]
                documents = list(collection.find()) # Fetch all documents
                print(f"DEBUG: Found {len(documents)} documents in MongoDB collection '{coll_name}'.")

                if build_func:
                    for doc_num, doc in enumerate(documents):
                        doc_id = doc.get('_id', f'MISSING_ID_{doc_num}')
                        text = build_func(doc)
                        # print(f"DEBUG: [{coll_name}] Doc ID: {doc_id} -> Text: '{text[:100]}...'") # Optional detailed logging
                        if text and text.strip():
                            text_blocks.append(text)
                            if coll_name in ["students", "professors"]:
                                dept_val = doc.get("department_id", "").strip()
                                metadatas.append({"department": dept_val})
                        else:
                            print(f"WARN: [{coll_name}] Skipping empty/invalid text block for Doc ID: {doc_id}")
                else:
                     print(f"WARN: No build function found for collection '{coll_name}'. Skipping text generation.")

                print(f"DEBUG: [{coll_name}] Total valid text blocks generated: {len(text_blocks)}")

                if text_blocks:
                    vector_store = None
                    if coll_name in ["students", "professors"]:
                        # Create index with metadata for students/professors
                        if metadatas and len(metadatas) == len(text_blocks):
                             print(f"DEBUG: Creating FAISS index for '{coll_name}' with metadata...")
                             vector_store = FAISS.from_texts(text_blocks, embedding_model, metadatas=metadatas)
                        else:
                             print(f"ERROR: Mismatch between text blocks ({len(text_blocks)}) and metadatas ({len(metadatas)}) for '{coll_name}'. Indexing without metadata.")
                             vector_store = FAISS.from_texts(text_blocks, embedding_model) # Fallback
                    elif coll_name == "departments" and splitter:
                         # Combine, split, and index department data (if splitter provided)
                         print(f"DEBUG: Splitting and creating FAISS index for '{coll_name}'...")
                         combined_text = "\n\n".join(text_blocks) # Join with separation
                         chunks = splitter.split_text(combined_text)
                         print(f"DEBUG: Split '{coll_name}' into {len(chunks)} chunks.")
                         if chunks:
                             vector_store = FAISS.from_texts(chunks, embedding_model)
                    else:
                         # Index other collections without metadata or specific splitting
                         print(f"DEBUG: Creating FAISS index for '{coll_name}' without specific handling...")
                         vector_store = FAISS.from_texts(text_blocks, embedding_model)

                    if vector_store:
                        # Ensure directory exists before saving
                        os.makedirs(index_dir, exist_ok=True)
                        vector_store.save_local(index_dir)
                        print(f"DEBUG: Index for '{coll_name}' created and saved to '{index_dir}'.")
                    else:
                        print(f"WARN: Vector store was not created for '{coll_name}'.")
                else:
                    print(f"DEBUG: No text data generated for '{coll_name}'. Skipping index creation.")

            except Exception as e:
                st.error(f"Error creating index for '{coll_name}': {e}")
                print(f"ERROR: Error creating index for '{coll_name}': {e}")
                traceback.print_exc()

        # --- Try loading the index (either newly created or pre-existing) ---
        try:
            if os.path.exists(index_path):
                print(f"DEBUG: Loading index for '{coll_name}' from '{index_dir}'...")
                # Ensure dangerous deserialization is allowed if needed (standard for FAISS local loading)
                vector_stores[coll_name] = FAISS.load_local(
                    index_dir,
                    embedding_model,
                    allow_dangerous_deserialization=True
                )
                print(f"DEBUG: Successfully loaded index for '{coll_name}'.")
            else:
                 print(f"WARN: Index file '{index_path}' not found after creation attempt. Cannot load index for '{coll_name}'.")
                 vector_stores[coll_name] = None
        except Exception as e:
            st.error(f"Error loading index for '{coll_name}': {e}")
            print(f"ERROR: Failed to load index for '{coll_name}': {e}")
            traceback.print_exc()
            vector_stores[coll_name] = None

    # Close MongoDB connection after processing all collections
    try:
        client.close()
        print("DEBUG: MongoDB connection closed.")
    except Exception as e:
        print(f"WARN: Error closing MongoDB connection: {e}")

    print("DEBUG: Finished index creation/loading process.")
    return vector_stores