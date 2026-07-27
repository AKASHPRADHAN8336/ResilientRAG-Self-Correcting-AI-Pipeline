import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rag.vector_store import get_vector_store

def ingest_pdf(file_path: str):
    """
    Loads a PDF, splits it into semantic chunks, and saves it to ChromaDB.
    """
    print(f"📄 Loading document: {file_path}")
    
    try:
        loader = PyPDFLoader(file_path)
        docs = loader.load()
    except Exception as e:
        print(f"Error loading PDF: {e}")
        return
    print("Chunking text...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = text_splitter.split_documents(docs)
    
    print(f"Saving {len(chunks)} chunks to ChromaDB...")
    db = get_vector_store()
    
    # Add the chunks to the database
    db.add_documents(chunks)
    
    print("Ingestion complete! The database is ready.")

if __name__ == "__main__":
    # Create the documents folder if it doesn't exist
    os.makedirs("documents", exist_ok=True)
    
    # Define the path to our sample PDF
    sample_pdf_path = "data_1.pdf"
    
    # Check if the user actually put a PDF there
    if os.path.exists(sample_pdf_path):
        ingest_pdf(sample_pdf_path)
    else:
        print(f"No PDF found at '{sample_pdf_path}'.")
        print("Please place a sample PDF in the 'documents' folder and run this script again.")
