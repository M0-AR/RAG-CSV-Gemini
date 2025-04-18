import streamlit as st
import os
from langchain_community.document_loaders import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Check if Google API key is available
if os.environ.get("GOOGLE_API_KEY") is None:
    st.error("GOOGLE_API_KEY not found in environment variables. Please ensure it's set in the .env file.")
    st.stop()

# Initialize Sentence Transformer embeddings
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

def main():
    st.set_page_config(page_title="Generate Embeddings for CSV", page_icon=":books:")
    st.title("Generate Embeddings for CSV Data")

    csv_file_path = "C:\\src\\Wateen\\ai-business-matcher\\ai_tools_detailed.csv"
    persist_directory = "./chroma_db_wateen"
    collection_name = "ai_tools_wateen"

    st.info(f"Reading data from: {csv_file_path}")
    st.info(f"Embeddings will be stored in: {persist_directory} (Collection: {collection_name})")

    if st.button("Generate Embeddings"):
        if not os.path.exists(csv_file_path):
            st.error(f"CSV file not found at: {csv_file_path}")
            return

        with st.spinner(f"Loading and processing {os.path.basename(csv_file_path)}..."):
            try:
                # Load the CSV file
                # Specify the column containing the text data if it's not the first one
                # Example: loader = CSVLoader(file_path=csv_file_path, csv_args={'delimiter': ','}, source_column='description')
                loader = CSVLoader(file_path=csv_file_path, encoding='utf-8') # Adjust encoding if needed
                documents = loader.load()

                if not documents:
                    st.warning("No documents were loaded from the CSV. Check the file content and format.")
                    return

                # Split documents into chunks
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                texts = text_splitter.split_documents(documents)

                if not texts:
                    st.warning("No text chunks were generated after splitting. Check document content.")
                    return

                # Create Chroma vector store
                st.write(f"Creating embeddings for {len(texts)} text chunks...")
                vectordb = Chroma.from_documents(
                    documents=texts,
                    embedding=embeddings,
                    persist_directory=persist_directory,
                    collection_name=collection_name
                )
                vectordb.persist() # Ensure data is saved
                st.success(f"Embeddings generated and saved to {persist_directory}!")

            except Exception as e:
                st.error(f"An error occurred: {e}")
                st.exception(e) # Show full traceback

if __name__ == '__main__':
    main()