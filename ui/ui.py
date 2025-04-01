import streamlit as st
from PIL import Image

def setup_sidebar():
    """
    Sets up the sidebar with:
      - A header for document and index management.
      - Instructions on uploading PDFs.
      - A file uploader for PDFs.
      - A button to trigger PDF vector store creation.
    """
    st.sidebar.header("Documents")
    st.sidebar.write("Upload PDFs to create a PDF vector store. Meanwhile, MongoDB indexes load automatically in the background.")
    uploaded_files = st.sidebar.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)
    create_vector_store_button = st.sidebar.button("Create PDF Vector Store")
    return uploaded_files, create_vector_store_button

def display_project_title_and_logo(logo_path):
    """
    Displays the project title and logo in a two-column layout.
    The logo is on the left, and the title is on the right.
    """
    logo = Image.open(logo_path)
    col1, col2 = st.columns([1, 8])
    with col1:
        st.image(logo, width=80)
    with col2:
        st.title("Levelwise RAG Chatbot")

def setup_sidebar_progress_bar():
    """
    Sets up a progress bar and a status message placeholder in the sidebar.
    Returns both the progress bar object and the placeholder so they can be updated.
    """
    progress_bar = st.sidebar.progress(0)
    status_placeholder = st.sidebar.empty()
    return progress_bar, status_placeholder

def show_vector_store_ready():
    """
    Displays a success message in the sidebar indicating that the PDF vector store is ready.
    """
    st.sidebar.success("PDF Vector Store is ready!")

def update_sidebar_status(message, status_type="info"):
    """
    Updates the sidebar status message.
    
    Parameters:
    - message: The text message to display.
    - status_type: One of "info", "warning", "success", or "error".
    """
    if status_type == "info":
        st.sidebar.info(message)
    elif status_type == "warning":
        st.sidebar.warning(message)
    elif status_type == "success":
        st.sidebar.success(message)
    elif status_type == "error":
        st.sidebar.error(message)
    else:
        st.sidebar.write(message)

def display_chat_history(messages):
    """
    Displays the entire chat history in the main area.
    
    Each message is rendered using Streamlit's chat_message container.
    """
    for msg in messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

def display_chat_message(role, content):
    """
    Displays a single chat message.
    
    Parameters:
    - role: "user" or "assistant"
    - content: The message content to display.
    """
    with st.chat_message(role):
        st.markdown(content)
