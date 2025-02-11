# Local RAG Chatbot with Langchain, DeepSeek-R1 and ChromaDB

This project is a Retrieval-Augmented Generation (RAG) chatbot built with Streamlit, ChromaDB, and LangChain. It provides responses based on document embeddings stored in ChromaDB for two knowledge bases: **Mobily** and **Caterpillar**.

## Setup Instructions

### Prerequisites

Ensure you have the following installed:

- Python 3.8+
- pip

### Install Dependencies

Run the following command to install the required dependencies:

```sh
pip install -r requirements.txt
```

### Setup ChromaDB

Ensure you have the ChromaDB directories available at:

- `./chroma_db_mobily`
- `./chroma_db_caterpillar`

If these directories do not exist, populate them with your document embeddings before running the chatbot.


## Creating Embeddings (Optional, as I have already provided embeddings)

If you want to create embeddings for a PDF, use the `embeddings.py` script:

```sh
streamlit run embeddings.py
```

This script processes PDF documents and generates embeddings that can be later used for retrieval. By default, it stores them in a separate ChromaDB folder (`./chroma_db`)


## Running the Application

Run the Streamlit application with:

```sh
streamlit run app.py
```


## Usage

1. Select a knowledge base (Mobily or Caterpillar).
2. Enter your query in the chat input field.
3. The chatbot will retrieve relevant documents and generate a concise response.

Enjoy using your RAG chatbot! 🚀

