import streamlit as st
import os
from PIL import Image  # Import for image handling
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()
groq_api_key = os.getenv('GROQ_API_KEY')
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Load the logo image
logo_path = r"D:\LevelWise_RAG\LevelWiseRAG.png"  # Update the path to your logo
logo = Image.open(logo_path)

# Set up the title and logo side by side
col1, col2 = st.columns([1, 8])  # Adjust the column width ratio if necessary
with col1:
    st.image(logo, width=80)  # Adjust width as necessary
with col2:
    st.title("Gemma Model Conversational RAG Chatbot")

# Initialize the Gemma model
llm = ChatGroq(groq_api_key=groq_api_key, model_name="gemma2-9b-it")

# Chat prompt template for conversation
prompt_template = ChatPromptTemplate.from_template(
"""
You are a conversational assistant. Answer the user's questions based on the provided documents and conversation context.
Please provide accurate and clear responses.
<context>
{context}
</context>
User's Question: {input}
"""
)

# Define the directory where uploaded files will be stored
UPLOAD_DIR = r"D:\LevelWise_RAG\research_papers"  # Update this path as needed

# Check if the directory exists, and create it if it doesn't
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

# Sidebar for uploading files and loading vector store
with st.sidebar:
    st.header("Upload Documents")
    uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)

    # Button to trigger document loading and vector store creation
    if st.button("Create Vector Store"):
        if uploaded_files:
            st.session_state.docs = []
            for uploaded_file in uploaded_files:
                # Create the file path for storing the file
                file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
                
                # Save the uploaded file to the defined directory
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Load the document
                loader = PyPDFLoader(file_path)
                st.session_state.docs.extend(loader.load())

            # Embed and create vector store
            st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
            st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)  # Split into chunks
            st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)  # Create vector store
            st.session_state.docs_loaded = True
            st.sidebar.success("Vector store created successfully!")
        else:
            st.sidebar.warning("Please upload files before creating the vector store.")

# Display a status if documents are loaded
if "docs_loaded" in st.session_state and st.session_state.docs_loaded:
    st.write("Vector store is ready!")

# Input for the user's question with chat history
if "messages" not in st.session_state:
    st.session_state.messages = []  # Initialize chat history

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input box for questions
if prompt := st.chat_input("Ask your question:"):
    # Display user input message in chat format
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if "vectors" in st.session_state:
        # Proceed with processing the user's question
        document_chain = create_stuff_documents_chain(llm, prompt_template)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        # Process user question with RAG and display the response
        start_time = time.process_time()
        response = retrieval_chain.invoke({'input': prompt})
        response_time = time.process_time() - start_time

        # Display model response in chat format
        answer = response['answer']
        st.session_state.messages.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.markdown(f"**Response time**: {response_time:.2f} seconds\n\n{answer}")

        # Display relevant document chunks
        with st.expander("Relevant Documents"):
            for i, doc in enumerate(response["context"]):
                st.write(f"**Document {i+1}:**")
                st.write(doc.page_content)
                st.write("--------------------------------")
    else:
        # If vector store is not initialized, inform the user
        assistant_message = "Please upload your documents and press 'Load Documents and Create Vector Store' before asking questions."
        st.session_state.messages.append({"role": "assistant", "content": assistant_message})
        with st.chat_message("assistant"):
            st.markdown(assistant_message)
