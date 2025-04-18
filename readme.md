# RAG Chatbot with LangChain, ChromaDB & Google Gemini

## Project Overview

A Retrieval-Augmented Generation (RAG) chatbot built using:

- **Streamlit** for the interactive UI
- **LangChain** for data loading, embeddings, and retrieval logic
- **ChromaDB** as a persistent vector store
- **Google Gemini LLM** via `langchain-google-genai` for high-quality responses
- **SentenceTransformerEmbeddings** via `langchain_community` for embedding generation

This application ingests AI tools data from a CSV file, generates embeddings, and provides an interactive chat interface to query the dataset of 3000+ AI tools.

## Features

- **Embeddings Generation:** Load CSV data, split into text chunks, and compute embeddings.
- **Vector Store Persistence:** Store and retrieve embeddings in ChromaDB (`./chroma_db_wateen`).
- **Interactive Chat Interface:** Streamlit-based chat app (`app.py`) for natural-language queries.
- **Google Gemini Integration:** Leverage the Gemini LLM for generating concise, informed answers.
- **Configurable Retrieval:** Adjustable `k` (number of retrieved chunks) and LLM `temperature` settings.
- **Environment Configuration:** Secure API keys and model settings via a `.env` file.

## Prerequisites

- **Python:** 3.8 or higher
- **pip:** Package installer for Python
- **Google API Key:** Access to Gemini models

## Setup Instructions

1. **Clone the Repository**

   ```bash
   git clone https://github.com/your-username/RAG-CSV-Gemini.git
   cd RAG-CSV-Gemini
   ```

2. **Create and Activate a Virtual Environment**

   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Configure Environment Variables**

   - Copy `.env.example` to `.env` (or create `.env` in the project root)
   - Add your Google API credentials:
     ```ini
     GOOGLE_API_KEY=your_google_api_key
     GOOGLE_API_MODEL=gemini-1.5-flash
     ```

4. **Install Dependencies**

   ```bash
   pip install --force-reinstall -r requirements.txt
   ```

## Usage

### Generate Embeddings

Run the embedding script to index your CSV dataset into ChromaDB:

```bash
streamlit run embeddings.py
```

- **Upload CSV Path:** Configured in `embeddings.py` (default: `ai_tools_detailed.csv`).
- **Output:** Embeddings persisted to `./chroma_db_wateen`.

### Start the Chat Application

Launch the Streamlit app for interactive Q&A:

```bash
streamlit run app.py
```

- Open your browser at `http://localhost:8501`.
- Ask questions about tool categories, counts, or specific features.

## Project Structure

```
RAG-CSV-Gemini/
├── .env                 # Environment variables (Google API keys)
├── requirements.txt     # Pinned dependencies for compatibility
├── embeddings.py        # Script to generate and persist embeddings
├── app.py               # Streamlit chat application
├── chroma_db_wateen/    # Persisted ChromaDB vector store
└── README.md            # Project documentation
```

## Dependencies & Compatibility

Key libraries and pinned versions (see `requirements.txt`):

- streamlit==1.29.0
- chromadb==0.6.3
- langchain==0.1.16
- langchain-core==0.1.52
- langchain_community>=0.0.32,<0.1.0
- langchain-google-genai==1.0.1
- sentence-transformers==4.1.0
- huggingface-hub==0.30.2
- pandas==2.2.1
- python-dotenv==1.0.0

## Contribution

Contributions, issues, and feature requests are welcome! Please open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [LangChain](https://github.com/langchain/langchain)
- [ChromaDB](https://github.com/chroma-core/chroma)
- [Google Gemini](https://developers.generativeai.google/)
- [Streamlit](https://streamlit.io)

## Inspiration
This project was inspired by: https://github.com/arzoodev/RAG-using-open-source-LLM.git
