import os
import shutil
import base64


TEXT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Testriq_info.txt")
DB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "testriq_db")


def load_text_content():
    """Load knowledge base content from local file or Streamlit secrets."""
    # Try local file first (for local development)
    if os.path.exists(TEXT_FILE):
        with open(TEXT_FILE, "r", encoding="utf-8") as f:
            content = f.read()
        print(f"Loaded from local file: {TEXT_FILE} ({len(content)} chars)")
        return content

    # Fall back to Streamlit secrets (base64-encoded for cloud deployment)
    try:
        import streamlit as st
        encoded = st.secrets["KNOWLEDGE_BASE"]
        content = base64.b64decode(encoded).decode("utf-8")
        print(f"Loaded from Streamlit secrets ({len(content)} chars)")
        return content
    except Exception as e:
        print(f"Secrets error: {e}")

    raise FileNotFoundError(
        "Knowledge base not found! Provide Testriq_info.txt locally "
        "or set KNOWLEDGE_BASE in Streamlit secrets."
    )


def build_knowledge_base():
    """Build ChromaDB vector store from Testriq_info.txt.
    
    Called automatically on first run if testriq_db/ doesn't exist.
    Returns the created vector store.
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import Chroma
    from langchain_community.embeddings import HuggingFaceEmbeddings

    # Delete old database if it exists
    if os.path.exists(DB_DIR):
        shutil.rmtree(DB_DIR)
        print("Old database deleted")

    # Load knowledge base content
    text_content = load_text_content()

    # Chunk the content
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=75,
        separators=["\n\n\n", "\n\n", "\n", ". ", " ", ""]
    )

    chunks = splitter.create_documents(
        [text_content],
        metadatas=[{"source": "official_knowledge_base", "priority": "high"}]
    )
    for chunk in chunks:
        chunk.metadata["source"] = "official_knowledge_base"
        chunk.metadata["priority"] = "high"

    print(f"Created {len(chunks)} chunks from knowledge base")

    # Create embeddings and store
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory=DB_DIR
    )
    print("Knowledge base created and saved!")
    return vectorstore


def db_exists():
    """Check if the ChromaDB vector store already exists."""
    return os.path.exists(DB_DIR) and os.path.exists(os.path.join(DB_DIR, "chroma.sqlite3"))


if __name__ == "__main__":
    print("=" * 60)
    print("TESTRIQ KNOWLEDGE BASE BUILDER")
    print("=" * 60)
    build_knowledge_base()
    print("\n" + "=" * 60)
    print("DONE! Knowledge base is ready.")
    print("=" * 60)