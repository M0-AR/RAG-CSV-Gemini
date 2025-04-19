import os
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.schema import Document

from supabase import create_client, Client
from langchain_community.vectorstores.supabase import SupabaseVectorStore
from dotenv import load_dotenv


def main():
    load_dotenv()
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_KEY = os.getenv("SUPABASE_KEY")
    TABLE_NAME = os.getenv("SUPABASE_TABLE", "ai_tools")  # Default table

    if not SUPABASE_URL or not SUPABASE_KEY:
        print("ERROR: Set SUPABASE_URL and SUPABASE_KEY in .env")
        return

    # Initialize Supabase client
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

    # 1) Load and chunk CSV
    csv_file_path = "C:\\src\\Wateen\\ai-business-matcher\\ai_tools_detailed.csv"
    df = pd.read_csv(csv_file_path)
    docs = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    for idx, row in df.iterrows():
        # Combine name and description for the main content
        text = f"{row.get('name', '')}: {row.get('description', '')}\n\nDetailed Description: {row.get('detailed_description', '')}"
        
        for i, chunk in enumerate(splitter.split_text(text)):
            # Use 'name' as unique id (fallback to row index)
            base_id = row.get('name') or str(idx)
            unique_id = f"{base_id}-{i}"
            
            # Create comprehensive metadata dictionary with all available fields
            metadata = {
                'id': unique_id,
                'name': str(row.get('name', '')),
                'description': str(row.get('description', '')),
                'category': str(row.get('category', '')),
                'pricing': str(row.get('pricing', '')),
                'url': str(row.get('url', '')),
                'upvotes': int(row.get('upvotes', 0)) if pd.notna(row.get('upvotes')) else 0,  
                'detailed_description': str(row.get('detailed_description', '')),
                'pricing_model': str(row.get('pricing_model', '')),
                'free_trial': bool(row.get('free_trial', False)),  
                'tags': str(row.get('tags', '')),
                'visit_link': str(row.get('visit_link', '')),
                'matts_pick': bool(row.get('matts_pick', False))  
            }
            
            # Remove empty string values and None values to keep metadata clean
            metadata = {k: v for k, v in metadata.items() if v not in [None, '']}
            
            docs.append(
                Document(
                    page_content=chunk,
                    metadata=metadata
                )
            )

    # 2) Compute embeddings
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    # 3) Upsert into Supabase vector store
    store = SupabaseVectorStore.from_documents(
        documents=docs,
        embedding=embeddings,
        client=supabase,
        table_name=TABLE_NAME,
    )

    print(f" Upserted {len(docs)} chunks into Supabase table: {TABLE_NAME}")


if __name__ == "__main__":
    main()
