from langchain.schema import HumanMessage, AIMessage
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain

def build_chat_history(messages):
    chat_history = []
    for msg in messages:
        if msg["role"] == "user":
            chat_history.append(HumanMessage(content=msg["content"]))
        else:
            chat_history.append(AIMessage(content=msg["content"]))
    return chat_history

def get_retrieval_chain(llm, retriever, current_user=None):
    # Build user context string if provided
    user_context = ""
    if current_user:
        user_context = f"User Role: {current_user.get('role')}, Username: {current_user.get('username')}"
        if current_user.get("department"):
            user_context += f", Department: {current_user.get('department')}"
    
    history_aware_retriever = create_history_aware_retriever(
        llm, 
        retriever,
        ChatPromptTemplate.from_messages([
            ("system", f"{user_context}\nGiven a chat history and the latest user question, formulate a standalone question."),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])
    )
    
    system_prompt = (
        f"{user_context}\nYou are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question.\n\n{{context}}"
    )
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    return create_retrieval_chain(history_aware_retriever, question_answer_chain)
