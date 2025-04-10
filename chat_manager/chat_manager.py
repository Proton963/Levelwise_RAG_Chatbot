from langchain.schema import HumanMessage, AIMessage
from langchain.chains import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from typing import Dict, Any, Tuple, List 

# Keep this function as is
def build_chat_history(messages: List[Dict[str, str]]) -> List[Any]:
    chat_history = []
    for msg in messages:
        if msg["role"] == "user":
            chat_history.append(HumanMessage(content=msg["content"]))
        else:
            chat_history.append(AIMessage(content=msg["content"]))
    return chat_history

# Function to get reusable prompt components
def get_prompt_components(current_user: Dict[str, Any]) -> Tuple[str, str]:
    """Builds user context string and combined instructions based on user role."""
    user_context_str = ""
    if current_user:
        user_context_str = f"User Role: {current_user.get('role')}, Username: {current_user.get('username')}"
        if current_user.get("department"):
            user_context_str += f", Department: {current_user.get('department')}"

    # Use the latest refined base instructions
    base_instructions = (
        "Answer strictly based on the provided context documents.\n"
        "If the question asks 'how many' or for a count, accurately count the relevant items based *only* on the documents provided in the context. State the count clearly in a complete sentence (e.g., 'Based on the provided documents, there are X students.' or 'There are Y professors in the provided context.').\n"
        "When asked to count specific items (like 'students' or 'professors'), first identify the documents in the context that match that type (e.g., starting with '[Student]' or '[Professor]') and then count only those identified documents.\n"
        "When listing names or items, YOU MUST present each item on a separate line. Start each line with a hyphen and a space ('- '). If relevant, include details like department in parentheses. Example format:\n- Name1 (Department1)\n- Name2 (Department2)\n- Name3 (Department3)"
    )
    role_specific_instructions = ""
    if current_user and current_user.get("role") == "HOD":
         role_specific_instructions = (
             "You are a Head of Department with access to the complete college data.\n"
             "Use all relevant information from the context to provide comprehensive answers."
         )
    # Add elif here for other role-specific instructions if needed

    combined_instructions = f"{role_specific_instructions}\n{base_instructions}".strip()
    return user_context_str, combined_instructions

# Function creates JUST the history-aware retriever part of the chain
def create_history_aware_retriever_chain_component(llm: Any, retriever: Any, user_context_str: str, combined_instructions: str) -> Any:
    """Creates the history-aware retriever chain object."""
    history_aware_prompt = ChatPromptTemplate.from_messages([
        # Use combined instructions for consistency
        ("system", f"{user_context_str}\n{combined_instructions}\nGiven the chat history and the latest user question, formulate a standalone question based on the chat history and follow up question."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])
    history_aware_retriever_chain = create_history_aware_retriever(llm, retriever, history_aware_prompt)
    return history_aware_retriever_chain

# Function creates JUST the final QA (stuff docs) part of the chain
def create_qa_chain_component(llm: Any, user_context_str: str, combined_instructions: str) -> Any:
    """Creates the question-answering (stuff documents) chain object."""
    # Add ONLY to reinforce using just the provided context
    qa_system_prompt = (
        f"{user_context_str}\n{combined_instructions}\n"
        "You are an assistant for question-answering tasks. Use ONLY the following pieces of retrieved context to answer the question. If the context is empty or insufficient answer based on chat history and your knowledge, state that the provided context does not contain the answer.\n\nContext:\n{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        MessagesPlaceholder(variable_name="chat_history", optional=True), # Make history optional here if context is primary
        ("human", "{input}")
    ])
    Youtube_chain = create_stuff_documents_chain(llm, qa_prompt)
    return Youtube_chain

