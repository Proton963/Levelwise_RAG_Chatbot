import streamlit as st
import os
from PIL import Image  # Import for image handling
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.schema import HumanMessage, AIMessage
from dotenv import load_dotenv
import time


# Load environment variables
load_dotenv()
groq_api_key = os.getenv('GROQ_API_KEY')
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Set up the title and logo
logo_path = r"D:\LevelWise_RAG\LevelWiseRAG.png"  # Update the path to your logo
logo = Image.open(logo_path)
col1, col2 = st.columns([1, 8])  # Adjust the column width ratio if necessary
with col1:
    st.image(logo, width=80)  # Adjust width as necessary
with col2:
    st.title("Gemma Model Conversational RAG Chatbot")

# Initialize the Gemma model
llm = ChatGroq(groq_api_key=groq_api_key, model_name="gemma2-9b-it")

# Define the directory where uploaded files will be stored
UPLOAD_DIR = r"D:\LevelWise_RAG\research_papers"  # Update this path as needed

if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

# Sidebar for uploading files and creating vector store
with st.sidebar:
    st.header("Upload Documents")
    uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)

    if st.button("Create Vector Store"):
        if uploaded_files:
            progress_bar = st.progress(0)  # Initialize the progress bar

            st.session_state.docs = []
            total_steps = len(uploaded_files) + 3  # Define total steps for progress
            current_step = 0

            for uploaded_file in uploaded_files:
                file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                loader = PyPDFLoader(file_path)
                st.session_state.docs.extend(loader.load())
                current_step += 1
                progress_bar.progress(current_step / total_steps)

            # Embedding the documents
            st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            current_step += 1
            progress_bar.progress(current_step / total_steps)

            # Splitting the documents into chunks
            st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
            st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)
            current_step += 1
            progress_bar.progress(current_step / total_steps)

            # Create the vector store
            st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
            st.session_state.docs_loaded = True
            current_step += 1
            progress_bar.progress(current_step / total_steps)
            st.sidebar.success("Vector store created successfully!")
        else:
            st.sidebar.warning("Please upload files before creating the vector store.")

# Ensure the vector store is ready
if "docs_loaded" not in st.session_state:
    st.session_state.docs_loaded = False

# Initialize chat history if not already
if "messages" not in st.session_state:
    st.session_state.messages = []  # Initialize chat history

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input box for questions (visible immediately)
user_input = st.chat_input("Ask your question:")

# If user input exists
if user_input:
    # Check if vector store (documents) is loaded
    if not st.session_state.docs_loaded:
        assistant_message = "Hi there! Please upload your document to start conversing."
        st.session_state.messages.append({"role": "assistant", "content": assistant_message})
        with st.chat_message("assistant"):
            st.markdown(assistant_message)
    else:
        # Append the user question to the chat history
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Building conversation history as a list of HumanMessage and AIMessage
        def build_chat_history():
            chat_history = []
            for msg in st.session_state.messages:
                if msg["role"] == "user":
                    chat_history.append(HumanMessage(content=msg["content"]))
                else:
                    chat_history.append(AIMessage(content=msg["content"]))
            return chat_history

        # History-aware retriever with proper chat history
        history_aware_retriever = create_history_aware_retriever(
            llm, 
            st.session_state.vectors.as_retriever(),
            ChatPromptTemplate.from_messages([
                ("system", "Given a chat history and the latest user question which might reference context in the chat history, formulate a standalone question."),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}")
            ])
        )
        
        system_prompt = (
            "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question."
            "\n\n{context}"
        )
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}")
            ]
        )
        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        # Build chat context from message history as a list of HumanMessage and AIMessage objects
        chat_history = build_chat_history()

        # Process user question with RAG and display the response
        response = rag_chain.invoke({'input': user_input, 'chat_history': chat_history})
        answer = response['answer']

        # Append the assistant's response to the chat history
        st.session_state.messages.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.markdown(answer)
