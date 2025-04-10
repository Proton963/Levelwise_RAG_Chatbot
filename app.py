import os
import datetime
import streamlit as st
import traceback
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
# from langchain_groq import ChatGroq
from langchain.retrievers import EnsembleRetriever
from mongo_handler.mongo_handler import load_or_create_faiss_indexes
from chat_manager.chat_manager import (
    build_chat_history, get_prompt_components, 
    create_history_aware_retriever_chain_component, create_qa_chain_component 
)
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
# from access_control.access_control import store_uploaded_file
from embedding_manager.embedding_manager import process_uploaded_file_to_faiss
from pydantic import SecretStr
from typing import Any, Dict    

# -----------------------------------------------------------------------------
# 1. Login Section: Show login page if not logged in.
# -----------------------------------------------------------------------------
logo_path = "D:/LevelWise_RAG/assets/LevelWiseRAG.png"  # Adjust if needed

if "logged_in" not in st.session_state or not st.session_state.logged_in:
    login_page(logo_path)
    st.stop()

# -----------------------------------------------------------------------------
# 2. Load Environment Variables
# -----------------------------------------------------------------------------
load_dotenv()
# groq_api_key = os.getenv("GROQ_API_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY")
mongo_uri = os.getenv("MONGO_URI")
mongo_db_name = os.getenv("MONGO_DB_NAME", "collegeDB")

if not google_api_key:
    st.error("GROQ_API_KEY environment variable not found. Please set it.")
    st.stop()
if not mongo_uri:
    st.error("MONGO_URI environment variable not found. Please set it.")
    st.stop()

# -----------------------------------------------------------------------------
# 3. Retrieve Current User and Display in Sidebar
# -----------------------------------------------------------------------------
current_user = st.session_state.current_user
display_user_info()

# -----------------------------------------------------------------------------
# 4. Initialize LLM, Embedding Model, and Text Splitter
# -----------------------------------------------------------------------------
# llm = ChatGroq(api_key=SecretStr(groq_api_key), model="gemma2-9b-it", stop_sequences=[])
llm = ChatGoogleGenerativeAI(api_key=SecretStr(google_api_key), model="gemini-1.5-flash")
embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)

# -----------------------------------------------------------------------------
# 5. Create and Load FAISS Indexes with Caching
# -----------------------------------------------------------------------------
# Define the collections you want to index
collections_to_index = ["departments", "professors", "students"]
print("DEBUG: Calling load_or_create_faiss_indexes from mongo_handler...")
all_mongodb_indexes = load_or_create_faiss_indexes(
    mongo_uri=mongo_uri,
    mongo_db_name=mongo_db_name,
    collections_to_index=collections_to_index,
    _embedding_model=embedding_model, # Use normal name in call
    _splitter=splitter              # Use normal name in call
)
print(f"DEBUG: Indexes returned from mongo_handler: {list(all_mongodb_indexes.keys())}")

# -----------------------------------------------------------------------------
# 6. Filter Data Sources Based on User Role (RBAC)
# -----------------------------------------------------------------------------
def select_indexes_for_user(all_indexes, user):
    role = user.get("role")
    if role == "HOD":
        return all_indexes
    elif role == "Professor":
        return {k: v for k, v in all_indexes.items() if k in ["professors", "students"]}
    elif role == "Student":
        return {k: v for k, v in all_indexes.items() if k in ["students", "professors"]}
    else:
        return {}

filtered_indexes = select_indexes_for_user(all_mongodb_indexes, current_user)
print("DEBUG: Filtered indexes based on user role:", filtered_indexes)

# -----------------------------------------------------------------------------
# 7. Build Retrievers with Department Filter (applied after caching)
# -----------------------------------------------------------------------------
print("DEBUG: --- Setting up Retrievers (Section 7) ---")
retrievers = [] # Initialize list to hold all retrievers
DEFAULT_RETRIEVER_K = 20 # Default K for broader MongoDB search
UPLOADED_DOC_K = 5 # K for uploaded doc search (can be different)

user_role = current_user.get("role")
user_dept = current_user.get("department")

print(f"DEBUG: Current User Role: {user_role}, Department: {user_dept}. Base K={DEFAULT_RETRIEVER_K}")

# --- Create Retrievers for MongoDB Indexes ---
print("DEBUG: Creating retrievers for MongoDB indexes...")
for key, vs in filtered_indexes.items():
    if vs is not None:
        # Initialize search_kwargs with default K
        # Note: Assuming HOD might need higher K, adjust if using single DEFAULT_RETRIEVER_K
        current_k = DEFAULT_RETRIEVER_K # Start with default
        # Example: If HOD needs more from MongoDB specifically
        # if user_role == "HOD":
        #    current_k = HOD_RETRIEVER_K # Defined elsewhere

        search_kwargs: Dict[str, Any] = {'k': current_k}

        # Determine if Filter needs to be applied (for non-HOD roles)
        apply_filter = False
        if user_role in ["Student", "Professor"] and key in ["students", "professors"]:
             if user_dept:
                 search_kwargs["filter"] = {"department": user_dept}
                 apply_filter = True
                 print(f"DEBUG: Applying department filter '{user_dept}' for role '{user_role}' on index '{key}'. Final kwargs: {search_kwargs}")
             else:
                 print(f"WARN: Role '{user_role}' needs department for filtering '{key}'. No filter applied. Kwargs: {search_kwargs}")
        else:
            # No filter for HOD or other combinations
            print(f"DEBUG: No department filter needed for role '{user_role}' on index '{key}'. Using kwargs: {search_kwargs}")

        # Create and add the retriever for this MongoDB index
        try:
            retriever = vs.as_retriever(search_kwargs=search_kwargs)
            retrievers.append(retriever)
            print(f"DEBUG: Successfully created retriever for MongoDB index '{key}'.")
        except Exception as e:
             print(f"ERROR: Failed to create retriever for MongoDB index '{key}': {e}")
             traceback.print_exc() # Added traceback for detail
# --- End MongoDB Retriever Loop ---


# --- ADDED: Create Retriever for Uploaded Document (If Exists in Session) ---
print("DEBUG: Checking for uploaded document vector store in session state...")
if "uploaded_doc_vs" in st.session_state and st.session_state.uploaded_doc_vs is not None:
    print("DEBUG: Found uploaded document VS. Creating its retriever...")
    try:
        # Create a retriever for the uploaded document's vector store
        uploaded_doc_retriever = st.session_state.uploaded_doc_vs.as_retriever(
            search_kwargs={'k': UPLOADED_DOC_K} # Use specific K for uploaded docs
        )
        # Add it to the list of retrievers
        retrievers.append(uploaded_doc_retriever)
        print(f"DEBUG: Successfully created and added retriever for uploaded document (k={UPLOADED_DOC_K}).")
    except Exception as e:
        print(f"ERROR: Failed to create retriever for uploaded document: {e}")
        traceback.print_exc() # Added traceback
else:
     print("DEBUG: No uploaded document vector store found in session state.")
# --- END ADDED ---


# --- Create the EnsembleRetriever ---
ensemble_retriever = None # Initialize as None
if not retrievers:
    # Only show error if NO retrievers were created at all
    st.error("No data sources available (MongoDB or Uploaded File). Cannot answer questions.")
    print("ERROR: No retrievers available to create EnsembleRetriever.")
else:
    # If there's at least one retriever, create the ensemble
    # Assign equal weights for simplicity, can be adjusted for relevance tuning
    weights = [1.0] * len(retrievers)
    try:
        ensemble_retriever = EnsembleRetriever(retrievers=retrievers, weights=weights)
        print(f"DEBUG: EnsembleRetriever created successfully with {len(retrievers)} retriever(s).")
    except Exception as e:
        st.error("Failed to create the combined retriever. Functionality may be limited.")
        print(f"ERROR: Failed to create EnsembleRetriever: {e}")
        traceback.print_exc()

# -----------------------------------------------------------------------------
# 8. Display Title/Logo on Main Page
# -----------------------------------------------------------------------------
display_project_title_and_logo(logo_path)

# -----------------------------------------------------------------------------
# 9. Sidebar: File Upload with Metadata Storage
# -----------------------------------------------------------------------------
st.sidebar.header("Document Q&A")
uploaded_file = st.sidebar.file_uploader(
     "Upload PDF, TXT, DOCX, CSV, XLSX",
     type=["pdf", "txt", "docx", "csv", "xlsx"],
     key="file_uploader_key" # Use a unique key
)

# Clear related session state if file is removed or changed
if uploaded_file is None:
    if "processed_file_id" in st.session_state:
         print("DEBUG: Clearing uploaded file session state as file is removed.")
         del st.session_state["processed_file_id"]
    if "uploaded_doc_vs" in st.session_state:
         del st.session_state["uploaded_doc_vs"]
         # Optional: Rerun if you want the retriever to update immediately on file removal
         # st.rerun()

if uploaded_file is not None:
    file_name = uploaded_file.name
    # Create a unique ID for the uploaded file instance to detect changes
    session_file_id = f"{file_name}_{uploaded_file.size}_{uploaded_file.type}"

    # Process only if it's a new file upload compared to what's processed in session
    if st.session_state.get("processed_file_id") != session_file_id:
        st.sidebar.info(f"Processing '{file_name}'...")
        print(f"DEBUG: New file detected, calling embedding_manager: {file_name}")
        try:
            # Ensure embedding_model and splitter are available (should be from Section 4)
            if 'embedding_model' not in locals() or embedding_model is None:
                 raise ValueError("Embedding model is not ready.")
            if 'splitter' not in locals() or splitter is None:
                 raise ValueError("Text splitter is not ready.")

            # Call the function from embedding_manager to handle all processing
            with st.spinner(f"Processing {file_name}... Please wait."):
                 faiss_index = process_uploaded_file_to_faiss(
                      uploaded_file,
                      embedding_model,
                      splitter
                 )

            if faiss_index:
                # Store the created index in session state
                st.session_state.uploaded_doc_vs = faiss_index
                # Mark this specific file instance as processed
                st.session_state.processed_file_id = session_file_id
                st.session_state.processed_file_name = file_name # Store name if needed for display
                st.sidebar.success(f"✅ '{file_name}' processed and ready!")
                print(f"DEBUG: Index for '{file_name}' received from embedding_manager.")
                # IMPORTANT: Rerun the script to update the ensemble_retriever (in Section 7)
                st.rerun()
            else:
                st.sidebar.error(f"Failed to process '{file_name}'. Check logs.")
                # Clear potentially outdated state
                st.session_state.uploaded_doc_vs = None
                st.session_state.processed_file_id = None

        except Exception as e:
            st.sidebar.error(f"Error during file processing: {e}")
            print(f"ERROR processing file '{file_name}': {e}")
            traceback.print_exc()
            # Clear state on error
            st.session_state.uploaded_doc_vs = None
            st.session_state.processed_file_id = None
    else:
        # File already processed in this session, do nothing or show confirmation
        st.sidebar.info(f"'{st.session_state.get('processed_file_name', 'File')}' is active for Q&A.")

# -----------------------------------------------------------------------------
# 10. Initialize Session State for Chat Messages
# -----------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "How can I help you?"}]

# -----------------------------------------------------------------------------
# 11. Main Chat Interface: Display Chat History and Process User Input
# -----------------------------------------------------------------------------
display_chat_history(st.session_state.messages)
user_input = st.chat_input("Ask your question:")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    display_chat_message("user", user_input)

    chat_history = []
    try:
        chat_history = build_chat_history(st.session_state.messages)
    except Exception as e:
        st.error(f"Error building chat history: {e}")
        print(f"ERROR: Could not build chat history. {e}")
        traceback.print_exc()

    # Check if essential components are available
    if ensemble_retriever is not None and llm is not None and current_user is not None:
        try:
            print("\nDEBUG: Starting modular RAG orchestration...")

            # 1. Get Prompt Components from Chat Manager
            user_context_str, combined_instructions = get_prompt_components(current_user)
            print("DEBUG: Prompt components loaded from chat_manager.")

            # 2. Create Chain Components from Chat Manager
            history_aware_retriever_chain = create_history_aware_retriever_chain_component(
                llm, ensemble_retriever, user_context_str, combined_instructions
            )
            qa_chain = create_qa_chain_component(
                llm, user_context_str, combined_instructions
            )
            print("DEBUG: Chain components created via chat_manager.")

            # 3. Run History-Aware Retrieval (using the component)
            print("DEBUG: Running history aware retrieval...")
            docs = history_aware_retriever_chain.invoke({"input": user_input, "chat_history": chat_history})
            print(f"DEBUG: History-aware retriever returned {len(docs)} documents.")

            # --- Optional: Keep retriever debug output here if needed ---
            # print("--- DEBUG: Unfiltered Retrieved Documents ---")
            # ... (loop to print docs - same as previous debug block) ...
            # print("--- END DEBUG: Unfiltered Retrieved Documents ---")
            # ----------------------------------------------------------

            # 4. Post-Retrieval Filtering (Logic stays in app.py for orchestration)
            filtered_docs = docs # Default to using all retrieved docs
            query_lower = user_input.lower()
            is_student_query = "student" in query_lower or "students" in query_lower
            is_prof_query = "professor" in query_lower or "professors" in query_lower

            # Apply filter only if the query is clearly about one type and not both
            if is_student_query and not is_prof_query:
                print("DEBUG: Filtering retrieved docs for '[Student]' content...")
                filtered_docs = [doc for doc in docs if doc.page_content.strip().startswith("[Student]")]
                print(f"DEBUG: Kept {len(filtered_docs)} student documents after filtering.")
            elif is_prof_query and not is_student_query:
                print("DEBUG: Filtering retrieved docs for '[Professor]' content...")
                filtered_docs = [doc for doc in docs if doc.page_content.strip().startswith("[Professor]")]
                print(f"DEBUG: Kept {len(filtered_docs)} professor documents after filtering.")
            else:
                # No specific filtering if query is ambiguous, general, or mentions both
                print("DEBUG: No specific student/professor filtering applied based on query keywords.")

            # 5. Handle No Documents Found After Filtering
            if not filtered_docs:
                print("WARN: No relevant documents found after retrieval and filtering.")
                # Check if original docs were also empty
                if not docs:
                    answer = "I couldn't find any relevant information to answer your question."
                else:
                    answer = "I found some general information, but nothing specific matching the exact type you asked for (e.g., students/professors)."
                # Skip QA chain invocation
            else:
                # 6. Run QA Chain with Filtered Context (using the component)
                print(f"DEBUG: Invoking QA chain with {len(filtered_docs)} documents...")
                # create_stuff_documents_chain returns the final answer string directly
                answer = qa_chain.invoke({
                    "input": user_input, # Pass original user input
                    "chat_history": chat_history, # Pass chat history for context
                    "context": filtered_docs # Pass the filtered documents
                })
                print(f"DEBUG: QA chain invocation complete. Answer preview: {answer[:100]}...")

            # 7. Display Answer
            st.session_state.messages.append({"role": "assistant", "content": answer})
            display_chat_message("assistant", answer)

        except Exception as e:
            st.error(f"An error occurred while processing your request. Please check logs.")
            print(f"ERROR during modular RAG chain execution: {e}")
            traceback.print_exc()

    else:
        # Handle missing essential components like retriever, llm, or user info
        missing = []
        if ensemble_retriever is None: missing.append("Retriever")
        if llm is None: missing.append("LLM")
        if current_user is None: missing.append("User Info")
        error_msg = f"Cannot process query due to missing components: {', '.join(missing)}"
        st.error(error_msg)
        print(f"ERROR: {error_msg}")
