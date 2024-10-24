import os
import streamlit as st
from document_handler.document_handler import load_documents
from embedding_manager.embedding_manager import create_embeddings_in_batches
from chat_manager.chat_manager import build_chat_history, get_retrieval_chain
from ui.ui import setup_sidebar, display_project_title_and_logo, setup_sidebar_progress_bar, show_vector_store_ready, display_chat_history, display_chat_message
from langchain_groq import ChatGroq
from dotenv import load_dotenv

# Load environment variables
# Load environment variables
load_dotenv()
groq_api_key = os.getenv('GROQ_API_KEY')
llm = ChatGroq(groq_api_key=groq_api_key, model_name="gemma2-9b-it")

# Display the project title and logo
logo_path = "D:/LevelWise_RAG/assets/LevelWiseRAG.png"  # Adjust the path if necessary
display_project_title_and_logo(logo_path)

# Set up the sidebar for document upload and vector store creation
uploaded_files, create_vector_store_button = setup_sidebar()

# Check if files are uploaded and create vector store when button is clicked
if uploaded_files and create_vector_store_button:
    progress_bar = setup_sidebar_progress_bar()
    
    # Process the documents
    documents = load_documents(uploaded_files, upload_dir="uploads")
    
    # Use batch and parallel processing to create embeddings
    batch_size = 5  # Define batch size based on document size and system capacity
    vector_store = create_embeddings_in_batches(documents, batch_size=batch_size, n_jobs=-1)  # Updated to use joblib-based function
    
    # Store the final vector store in session state
    st.session_state.vector_store = vector_store
    st.session_state.docs_loaded = True
    progress_bar.progress(100)  # Show full progress

    # Indicate that the vector store is ready
    show_vector_store_ready()

# Ensure the vector store is ready
if "docs_loaded" not in st.session_state:
    st.session_state.docs_loaded = False

# Initialize chat history with a welcome message if not already present
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({"role": "assistant", "content": "How can I help you?"})  # Add initial AI message

# Display chat history
display_chat_history(st.session_state.messages)

# User input box for questions (visible immediately)
user_input = st.chat_input("Ask your question:")

if user_input:
    if not st.session_state.docs_loaded:
        assistant_message = "Hi there! Please upload your document to start conversing."
        st.session_state.messages.append({"role": "assistant", "content": assistant_message})
        display_chat_message("assistant", assistant_message)
    else:
        # Append the user question to the chat history
        st.session_state.messages.append({"role": "user", "content": user_input})
        display_chat_message("user", user_input)

        # Build chat context from message history
        chat_history = build_chat_history(st.session_state.messages)

        # Get retrieval chain
        rag_chain = get_retrieval_chain(llm, st.session_state.vector_store)

        # Process user question with RAG and display the response
        response = rag_chain.invoke({'input': user_input, 'chat_history': chat_history})
        answer = response['answer']

        # Append the assistant's response to the chat history
        st.session_state.messages.append({"role": "assistant", "content": answer})
        display_chat_message("assistant", answer)



        
#D:/LevelWise_RAG/assets/LevelWiseRAG.png