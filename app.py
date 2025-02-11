import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
import re

# Constants
CHROMA_DB_PATHS = {
    "Mobily": "./chroma_db_mobily",
    "Caterpillar": "./chroma_db_caterpillar"
}
PROMPT_TEMPLATE = """
You are an expert research assistant. Use the provided context to answer the query. 
If unsure, state that you don't know. Be concise and factual (max 3 sentences).

Query: {user_query} 
Context: {document_context} 
Answer:
"""
EMBEDDING_MODEL = OllamaEmbeddings(model="deepseek-r1:1.5b")
LANGUAGE_MODEL = OllamaLLM(model="deepseek-r1:1.5b")


def get_chroma_instance(selected_db):
    """Load ChromaDB based on the selected database."""
    return Chroma(
        persist_directory=CHROMA_DB_PATHS[selected_db],
        collection_name="documents",
        embedding_function=EMBEDDING_MODEL
    )


def find_related_documents(chroma_db, query):
    """Retrieve relevant documents using similarity search."""
    return chroma_db.similarity_search(query, k=5)


def generate_answer(user_query, context_documents):
    """Generate AI response using retrieved documents."""
    context_text = "\n\n".join([doc.page_content for doc in context_documents])
    conversation_prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    response_chain = conversation_prompt | LANGUAGE_MODEL
    return response_chain.invoke({"user_query": user_query, "document_context": context_text})


def format_think_section(response):
    """
    Replaces all occurrences of '.\n' with '.*' only within the <think>...</think> tags.
    """
    def replace_first_last_with_star(s):
        """
        Replaces the first and last characters of a string with '*'.
        """
        if not s:  # Handle empty string
            return s
        # Replace first character
        s = "*" + s[1:]
        # Replace last character if the string has more than one character
        if len(s) > 1:
            s = s[:-1] + "*"
        return s


    def replace_dot_newline(match):
        content = match.group(1)  # Extract content inside <think>...</think>
        content = replace_first_last_with_star(content)
        modified_content = re.sub(r'\.\n\n', '.*\n\n*', content)  # Replace '.\n' with '.*'
        return f"<think>{modified_content}</think>"

    return re.sub(r"<think>(.*?)</think>", replace_dot_newline, response, flags=re.DOTALL)

# UI Configuration
st.title("📘 Test Task")
st.markdown("### RAG Chatbot for Mobily and Caterpillar PDFs")
st.markdown("##### Note: The text within the <think> tags shows the thought process of the RAG Chatbot")
st.markdown("---")

# Select ChromaDB
selected_db = st.selectbox("Select Knowledge Base", options=list(CHROMA_DB_PATHS.keys()))
chroma_db = get_chroma_instance(selected_db)

user_input = st.chat_input("Ask a question about your documents...")

if user_input:
    with st.chat_message("user"):
        st.write(user_input)

    with st.spinner(f"Searching in {selected_db}..."):
        relevant_docs = find_related_documents(chroma_db, user_input)
        ai_response = generate_answer(user_input, relevant_docs)
        ai_response = format_think_section(ai_response)

    with st.chat_message("assistant", avatar="🤖"):
        st.write(ai_response)
