import streamlit as st
from groq import Groq
import os

# ─── Page Config (MUST be first Streamlit command) ───
st.set_page_config(page_title="Testriq QA Assistant", layout="centered")


@st.cache_resource
def load_embeddings():
    """Load HuggingFace embeddings model — cached so it only loads ONCE."""
    from langchain_community.embeddings import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


@st.cache_resource
def load_vectorstore():
    """Load ChromaDB vector store — auto-builds if missing."""
    from langchain_community.vectorstores import Chroma
    from rag import build_knowledge_base, db_exists, DB_DIR

    # Auto-build the knowledge base if it doesn't exist
    if not db_exists():
        st.info("🔨 Building knowledge base for the first time... This may take a minute.")
        build_knowledge_base()

    embeddings = load_embeddings()
    return Chroma(
        persist_directory=DB_DIR,
        embedding_function=embeddings
    )


def get_api_key():
    """Get GROQ API key from st.secrets (cloud) or .env (local dev)."""
    # Try Streamlit secrets first (for cloud deployment)
    try:
        return st.secrets["GROQ_API_KEY"]
    except (FileNotFoundError, KeyError):
        pass
    # Fallback to environment variable (for local dev)
    from dotenv import load_dotenv
    load_dotenv()
    key = os.getenv("GROQ_API_KEY")
    if not key:
        st.error("❌ GROQ_API_KEY not found. Set it in Streamlit secrets or .env file.")
        st.stop()
    return key


# Initialize clients
client = Groq(api_key=get_api_key())

# Load resources with caching (first load: ~10-15s, subsequent: instant)
with st.spinner("Loading knowledge base..."):
    vectorstore = load_vectorstore()

# ─── Custom CSS for better UX ───
st.markdown("""
<style>
    .stApp {
        max-width: 900px;
        margin: 0 auto;
    }
    .main-header {
        text-align: center;
        padding: 1rem 0;
    }
    .main-header h1 {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.2rem;
        font-weight: 800;
    }
    .main-header p {
        color: #888;
        font-size: 0.9rem;
    }
    div[data-testid="stChatMessage"] {
        border-radius: 12px;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# ─── Header ───
st.markdown("""
<div class="main-header">
    <h1>Testriq QA Assistant</h1>
    <p>Powered by Groq + Llama 3.3 + Testriq Knowledge Base</p>
</div>
""", unsafe_allow_html=True)

# ─── System Prompt (intentionally loose for POC testing) ───
system_prompt = {
    "role": "system",
    "content": """You are a helpful QA Assistant for Testriq QA Lab, 
    a professional software testing company. Answer questions using 
    the context provided from Testriq's knowledge base. If the context doesn't 
    contain enough information, use your general QA knowledge but always 
    stay relevant to Testriq's services and expertise. Be professional, 
    clear, and helpful. Always present Testriq in a positive light."""
}

# ─── Session State ───
if "messages" not in st.session_state:
    st.session_state.messages = [system_prompt]

# ─── Display Previous Messages ───
for message in st.session_state.messages:
    if message["role"] == "system":
        continue
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ─── User Input ───
user_input = st.chat_input("Ask me anything about Testriq...")

if user_input:
    # Show user message immediately
    with st.chat_message("user"):
        st.markdown(user_input)

    # Save user message
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Show thinking spinner while processing
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Search knowledge base for relevant context
            docs = vectorstore.similarity_search(user_input, k=5)
            context = "\n\n".join([doc.page_content for doc in docs])

            # Build prompt with context
            rag_prompt = f"""Use the following information from Testriq's knowledge base to answer the question.

Context:
{context}

Question: {user_input}

Instructions:
- Answer based on the context above.
- If the context contains the answer, use it directly and cite the facts.
- If the context doesn't have enough info, use your general knowledge about QA and software testing.
- Keep your responses concise and well-structured.
- Use bullet points or numbered lists when listing multiple items."""

            # Build messages for API call (keep last 10 messages to avoid token limit)
            recent_messages = st.session_state.messages[-10:]
            messages_with_context = [system_prompt] + recent_messages[:-1] + [
                {"role": "user", "content": rag_prompt}
            ]

            # Get response from Groq
            try:
                response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=messages_with_context,
                    temperature=0.7,
                    max_tokens=1024
                )
                reply = response.choices[0].message.content
            except Exception as e:
                reply = f"⚠️ Error getting response: {str(e)}"

        # Display response
        st.markdown(reply)

    # Save bot response
    st.session_state.messages.append({"role": "assistant", "content": reply})