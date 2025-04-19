import streamlit as st
import os
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from supabase import create_client
from langchain_community.vectorstores.supabase import SupabaseVectorStore

# Load environment variables
load_dotenv()

google_api_key = os.environ.get("GOOGLE_API_KEY")
google_api_model = os.environ.get("GOOGLE_API_MODEL", "gemini-1.5-flash") # Default model if not set
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
SUPABASE_TABLE = os.environ.get("SUPABASE_TABLE", "ai_tools")

if not google_api_key:
    st.error("GOOGLE_API_KEY not found. Please ensure it's set in the .env file.")
    st.stop()

if not SUPABASE_URL or not SUPABASE_KEY:
    st.error("SUPABASE_URL and SUPABASE_KEY must be set in .env")
    st.stop()

# --- Configuration ---
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Initialize Supabase client
supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)

# --- Initialize Components ---
st.set_page_config(page_title="Chat with AI Tools CSV Data", page_icon=":robot_face:")
st.title("Chat with AI Tools Data")
st.write("Ask questions about the AI tools detailed in the CSV file[https://github.com/M0-AR/ai-business-matcher/blob/main/ai_tools_detailed.csv].")

@st.cache_resource # Cache resources for efficiency
def load_components():
    try:
        # Embeddings
        embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL_NAME)

        # Vector Store (Supabase)
        vectorstore = SupabaseVectorStore(
            client=supabase_client,
            embedding=embeddings,
            table_name=SUPABASE_TABLE
        )
        
        retriever = vectorstore.as_retriever(search_kwargs={'k': 3300}) # Retrieve top 1000 relevant chunks

        # LLM (Google Gemini)
        llm = ChatGoogleGenerativeAI(model=google_api_model, google_api_key=google_api_key, temperature=0.1)

        # Prompt Template
        template = """
You are an assistant for question-answering tasks about AI tools from a comprehensive database of over 3300 AI tools.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know.

For questions about specific categories of tools (like marketing, coding, etc.), try to provide a comprehensive answer.
For questions about counts or statistics, be honest about what you can determine from the context provided.

Question: {question}

Context: {context}

Answer:
"""

        prompt = ChatPromptTemplate.from_template(template)

        return retriever, llm, prompt
    except Exception as e:
        st.error(f"Error loading components: {e}")
        st.exception(e)
        return None, None, None

retriever, llm, prompt = load_components()

if not all([retriever, llm, prompt]):
    st.warning("Failed to initialize necessary components. Cannot proceed.")
    st.stop()

# --- RAG Chain Definition ---
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# --- Chat Interface ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if query := st.chat_input("Ask a question about the AI tools:"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": query})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(query)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = rag_chain.invoke(query)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                error_msg = f"An error occurred while processing your request: {e}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": f"Error: {error_msg}"})