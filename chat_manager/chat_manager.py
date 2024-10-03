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

def get_retrieval_chain(llm, vector_store):
    history_aware_retriever = create_history_aware_retriever(
        llm, 
        vector_store.as_retriever(),
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
    return create_retrieval_chain(history_aware_retriever, question_answer_chain)
