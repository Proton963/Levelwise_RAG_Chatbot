import streamlit as st
import base64
from PIL import Image

# Dummy credentials for testing login.
dummy_users = {
    "Rohit Jain": {"password": "password123", "role": "Student", "department": "CSE"},
    "Ishaan Gupta": {"password": "password123", "role": "Student", "department": "ECE"},
    "Prateek Mishra": {"password": "password123", "role": "Student", "department": "ME"},
    "Manish Kumar": {"password": "profpass", "role": "Professor", "department": "ME"},
    "Sneha Kulkarni": {"password": "profpass", "role": "Professor", "department": "CSE"},
    "Amit Sharma": {"password": "hodpass", "role": "HOD", "department": "CSE"},
    "Priya Verma": {"password": "hodpass", "role": "HOD", "department": "ECE"},
    "Rajesh Singh": {"password": "hodpass", "role": "HOD", "department": "ME"}
}

def _embed_logo_and_title_top_left(logo_path: str, title_text: str = "Levelwise RAG Chatbot"):
    """
    Returns an HTML string that places a Base64-encoded logo and a title
    in the top-left corner with minimal padding.
    """
    # Convert the local image file to base64 so we can embed it directly in HTML.
    with open(logo_path, "rb") as f:
        logo_bytes = f.read()
    logo_b64 = base64.b64encode(logo_bytes).decode()

    # HTML snippet with minimal top margin, left-aligned
    # Adjust margin or padding as needed for your design.
    html_code = f"""
    <div style="display: flex; align-items: center; margin-bottom: 1rem;">
        <img src="data:image/png;base64,{logo_b64}" style="width:80px; margin-right: 10px;" />
        <h2 style="margin: 0; padding: 0;">{title_text}</h2>
    </div>
    """
    return html_code

def login_page(logo_path: str):
    """
    Displays the login page with the chatbot logo and title in the top-left corner,
    followed by username/password inputs and a login button.
    """
    # Reduce Streamlit's default top padding for the page
    st.markdown(
        """
        <style>
        /* Remove or reduce top padding */
        .main .block-container {
            padding-top: 0rem !important;
            margin-top: 5rem !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Embed the top-left logo/title
    st.markdown(_embed_logo_and_title_top_left(logo_path), unsafe_allow_html=True)

    st.markdown("### Login")
    st.write("Please log in to access the chatbot.")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username in dummy_users and dummy_users[username]["password"] == password:
            st.session_state.logged_in = True
            st.session_state.current_user = {"username": username, **dummy_users[username]}
            st.success(f"Welcome, {username}!")
            st.rerun()  # Reload the app after login.
        else:
            st.error("Invalid username or password.")

def display_user_info():
    """Displays the current user's username and department at the top of the sidebar."""
    if "current_user" in st.session_state:
        user = st.session_state.current_user
        st.sidebar.markdown(f"**User:** {user['username']}  \n**Department:** {user['department']}")
    else:
        st.sidebar.markdown("**Not logged in**")


def display_marks_upload_section_sidebar():
    """
    Displays the marks file upload section in the sidebar (only for Professors/HODs).
    Returns a tuple (uploaded_file, subject) if the upload button is clicked,
    otherwise returns None.
    """
    if "current_user" not in st.session_state:
        st.error("User not logged in.")
        return None

    user = st.session_state.current_user
    if user.get("role") not in ["Professor", "HOD"]:
        st.sidebar.warning("You are not authorized to upload marks files.")
        return None

    st.sidebar.header("Upload Marks File")
    st.sidebar.markdown("Please provide the marks file along with the subject name.(For multiple subjects, separate with commas)")

    subject = st.sidebar.text_input("Subject for the marks file", key="marks_subject_sidebar")
    marks_file = st.sidebar.file_uploader(
        "Upload marks file to Database", 
        type=["pdf", "xlsx", "csv", "docx"],
        key="marks_file_sidebar"
    )

    if st.sidebar.button("Upload Marks File", key="marks_upload_button_sidebar"):
        if marks_file is None:
            st.sidebar.error("No file selected! Please choose a marks file to upload.")
            return None
        if not subject.strip():
            st.sidebar.error("Please enter a subject for the marks file.")
            return None
        return marks_file, subject
    
    return None

def setup_sidebar():
    st.sidebar.header("Documents")
    st.sidebar.write("Upload documents to create a PDF vector store.")
    uploaded_files = st.sidebar.file_uploader(
        "Choose a document", 
        type=["pdf", "txt", "xlsx", "doc", "csv"], 
        accept_multiple_files=False
    )
    create_vector_store_button = st.sidebar.button("Create PDF Vector Store")
    return uploaded_files, create_vector_store_button

def setup_sidebar_progress_bar():
    progress_bar = st.sidebar.progress(0)
    status_placeholder = st.sidebar.empty()
    return progress_bar, status_placeholder

def show_vector_store_ready():
    st.sidebar.success("PDF Vector Store is ready!")

def update_sidebar_status(message, status_type="info"):
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

def display_project_title_and_logo(logo_path):
    """
    Displays the project title and logo in a top-left corner style on the main screen,
    using Base64 embedding. Called after login if you want the same look as on the login page.
    """
    # Convert the local image file to base64 so we can embed it directly in HTML.
    with open(logo_path, "rb") as f:
        logo_bytes = f.read()
    logo_b64 = base64.b64encode(logo_bytes).decode()

    st.markdown(
        f"""
        <div style="display: flex; align-items: center; margin-bottom: 1rem;">
            <img src="data:image/png;base64,{logo_b64}" style="width:80px; margin-right: 10px;" />
            <h2 style="margin: 0; padding: 0;">Levelwise RAG Chatbot</h2>
        </div>
        """,
        unsafe_allow_html=True
    )

def display_chat_history(messages):
    for msg in messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

def display_chat_message(role, content):
    with st.chat_message(role):
        st.markdown(content)
