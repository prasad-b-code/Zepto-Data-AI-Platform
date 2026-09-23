from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions

# Define paths relative to this file
BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = BASE_DIR / "docs"
CHROMA_DIR = BASE_DIR / "chroma_db"
COLLECTION_NAME = "zepto_policy_collection"

def get_chroma_client():
    """Initializes and returns a persistent ChromaDB client."""
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(CHROMA_DIR))

def get_embedding_function():
    """Configures local sentence-transformers with all-MiniLM-L6-v2."""
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

def build_index(reset: bool = False):
    """
    Loads all 8 documents, creates a per-document chunk,
    embeds them with all-MiniLM-L6-v2, and stores them in ChromaDB.
    """
    client = get_chroma_client()
    emb_fn = get_embedding_function()

    # If resetting, delete any prior collection
    if reset:
        try:
            client.delete_collection(name=COLLECTION_NAME)
            print(f"[ChromaDB] Reset existing collection '{COLLECTION_NAME}'.")
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=emb_fn,
        metadata={"hnsw:space": "cosine"}
    )

    doc_files = sorted(DOCS_DIR.glob("doc_*.txt"))
    if not doc_files:
        raise FileNotFoundError(f"No doc_*.txt files found in {DOCS_DIR}")

    documents = []
    ids = []
    metadatas = []

    print(f"\n--- Loading and Chunking Documents from {DOCS_DIR} ---")
    for doc_path in doc_files:
        with open(doc_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
        
        doc_id = doc_path.stem  # e.g., 'doc_01'
        documents.append(content)
        ids.append(doc_id)
        metadatas.append({"source": doc_path.name, "doc_id": doc_id})
        print(f"Loaded: {doc_path.name} ({len(content)} characters)")

    # Insert / Upsert into ChromaDB
    print(f"\nEmbedding chunks with all-MiniLM-L6-v2 and indexing into '{COLLECTION_NAME}'...")
    collection.upsert(
        documents=documents,
        ids=ids,
        metadatas=metadatas
    )

    count = collection.count()
    print(f"[Success] ChromaDB collection '{COLLECTION_NAME}' now contains {count} indexed chunks.")
    return collection

if __name__ == "__main__":
    build_index(reset=True)