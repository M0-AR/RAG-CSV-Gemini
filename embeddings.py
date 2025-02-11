import streamlit as st
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
import chromadb
from chromadb.utils import embedding_functions


PDF_STORAGE_PATH = 'document_store/pdfs/'
EMBEDDING_MODEL = OllamaEmbeddings(model="deepseek-r1:1.5b")
CHROMA_PERSIST_DIR = "./chroma_db"

if "processed_docs" not in st.session_state:
    st.session_state.processed_docs = False

def save_uploaded_file(uploaded_file):
    file_path = PDF_STORAGE_PATH + uploaded_file.name
    with open(file_path, "wb") as file:
        file.write(uploaded_file.getbuffer())
    return file_path

def load_pdf_documents(file_path):
    document_loader = PDFPlumberLoader(file_path)
    return document_loader.load()

def chunk_documents(docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(docs)

class OllamaChromaEmbedder:
    def __init__(self, ollama_embeddings: OllamaEmbeddings):
        self.embeddings = ollama_embeddings
        
    def __call__(self, input):
        return self.embeddings.embed_documents(input)

def index_documents(document_chunks):
    Chroma.from_documents(
        documents=document_chunks,
        embedding=EMBEDDING_MODEL,
        persist_directory=CHROMA_PERSIST_DIR,
        collection_name="documents"
    )

# UI Configuration
st.title("📘 Task")
st.markdown("### PDF Embedding Generator")
st.markdown("---")

# File Upload Section
uploaded_pdf = st.file_uploader(
    "Upload PDF Document",
    type="pdf",
    help="Select a PDF document to generate embeddings",
    accept_multiple_files=False
)

if uploaded_pdf and not st.session_state.processed_docs:
    saved_path = save_uploaded_file(uploaded_pdf)
    raw_docs = load_pdf_documents(saved_path)
    processed_chunks = chunk_documents(raw_docs)
    
    # Store documents in ChromaDB
    index_documents(processed_chunks)
    
    st.session_state.processed_docs = True
    st.success("✅ Document processed successfully! Embeddings created and stored.")