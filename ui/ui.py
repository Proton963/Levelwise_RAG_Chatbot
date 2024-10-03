import streamlit as st
from PIL import Image

# Function to set up the sidebar for file uploads
def setup_sidebar():
    st.sidebar.header("Upload Documents")
    uploaded_files = st.sidebar.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)
    create_vector_store_button = st.sidebar.button("Create Vector Store")
    return uploaded_files, create_vector_store_button

# Function to display the project title and logo
def display_project_title_and_logo(logo_path):
    # Load the logo image
    logo = Image.open(logo_path)
    # Create two columns for layout
    col1, col2 = st.columns([1, 8])
    with col1:
        st.image(logo, width=80)  # Adjust logo width if needed
    with col2:
        st.title("Levelwise RAG Chatbot")

# Function to set up the progress bar in the sidebar
def setup_sidebar_progress_bar():
    progress_bar = st.sidebar.progress(0)  # Initialize progress bar
    return progress_bar

# Function to display that the vector store is ready
def show_vector_store_ready():
    st.sidebar.success("Vector store is ready!")

# Function to display the chat history
def display_chat_history(messages):
    for msg in messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

# Function to display a single chat message
def display_chat_message(role, content):
    with st.chat_message(role):
        st.markdown(content)
