import chromadb
import os

def init_chroma_db():
    chroma_path = os.getenv("CHROMA_PERSIST_DIRECTORY", "./data/chroma_db")
    
    # ensure the directory is correct
    os.makedirs(chroma_path, exist_ok=True)

    try:
        client = chromadb.PersistentClient(path=chroma_path)
        
        # collection: init documents level
        try:
            collection = client.get_collection(name="documents")
            print(f"Collection 'documents' already exists")
        except Exception:
            collection = client.create_collection(
                name="documents",
                metadata={
                    "hnsw:space": "cosine",
                    "dimension": 1536
                }
            )
            print(f"Successfully created collection 'documents'")
            
        # collection: init chunk level
        try:
            collection = client.get_collection(name="document_chunks")
            print(f"Collection 'document_chunks' already exists")
        except Exception:
            collection = client.create_collection(
                name="document_chunks",
                metadata={
                    "hnsw:space": "cosine",
                    "dimension": 1536
                }
            )
            print(f"Successfully created collection 'document_chunks'")
        
        print(f"ChromaDB initialized at {chroma_path}")
    except Exception as e:
        raise e
        print(f"An error occurred while initializing ChromaDB: {e}")
        # no exception for following process
        # ChromaRepository will create collection if needed

if __name__ == "__main__":
    init_chroma_db()