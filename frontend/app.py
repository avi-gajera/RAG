import streamlit as st
import requests
import time

# ─────────────────────── Config ───────────────────────
API_URL = "http://localhost:8001"

st.set_page_config(
    page_title="RAG System · Intelligent Document Q&A",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────── Custom CSS ───────────────────────
st.markdown("""
<style>
    /* ── Import Google Font ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* ── Global ── */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ── Main container ── */
    .main .block-container {
        padding-top: 2rem;
        max-width: 900px;
    }

    /* ── Hero header ── */
    .hero-header {
        text-align: center;
        padding: 2rem 0 1.5rem 0;
    }
    .hero-header h1 {
        font-size: 2.4rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
    }
    .hero-header p {
        color: #94a3b8;
        font-size: 1.05rem;
        font-weight: 300;
    }

    /* ── Chat messages ── */
    .chat-message {
        padding: 1.2rem 1.5rem;
        border-radius: 16px;
        margin-bottom: 1rem;
        line-height: 1.65;
        font-size: 0.95rem;
        animation: fadeIn 0.3s ease-in;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(8px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .user-msg {
        background: linear-gradient(135deg, #667eea22 0%, #764ba222 100%);
        border-left: 4px solid #667eea;
    }
    .bot-msg {
        background: rgba(30, 41, 59, 0.5);
        border-left: 4px solid #22d3ee;
    }

    /* ── Source chips ── */
    .source-chip {
        display: inline-block;
        background: linear-gradient(135deg, #667eea33 0%, #764ba233 100%);
        color: #c4b5fd;
        padding: 0.3rem 0.75rem;
        border-radius: 20px;
        font-size: 0.78rem;
        margin: 0.2rem 0.3rem 0.2rem 0;
        font-weight: 500;
        border: 1px solid #667eea44;
    }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e1b4b 100%);
    }
    section[data-testid="stSidebar"] .block-container {
        padding-top: 1.5rem;
    }

    .sidebar-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #e2e8f0;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* ── Document card ── */
    .doc-card {
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 12px;
        padding: 0.9rem 1rem;
        margin-bottom: 0.6rem;
        transition: all 0.2s ease;
    }
    .doc-card:hover {
        border-color: rgba(99, 102, 241, 0.5);
        transform: translateX(3px);
    }
    .doc-card .doc-name {
        color: #e2e8f0;
        font-weight: 500;
        font-size: 0.88rem;
        margin-bottom: 0.25rem;
    }
    .doc-card .doc-meta {
        color: #64748b;
        font-size: 0.75rem;
    }

    /* ── Status pill ── */
    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.35rem 0.85rem;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 500;
    }
    .status-online {
        background: rgba(34, 197, 94, 0.15);
        color: #4ade80;
        border: 1px solid rgba(34, 197, 94, 0.3);
    }
    .status-offline {
        background: rgba(239, 68, 68, 0.15);
        color: #f87171;
        border: 1px solid rgba(239, 68, 68, 0.3);
    }

    /* ── Upload area ── */
    [data-testid="stFileUploader"] {
        border: 2px dashed rgba(99, 102, 241, 0.3);
        border-radius: 12px;
        padding: 0.5rem;
        transition: border-color 0.3s ease;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: rgba(99, 102, 241, 0.6);
    }

    /* ── Chat input ── */
    [data-testid="stChatInput"] textarea {
        border-radius: 12px !important;
        border: 1px solid rgba(99, 102, 241, 0.3) !important;
        font-family: 'Inter', sans-serif !important;
    }
    [data-testid="stChatInput"] textarea:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.15) !important;
    }

    /* ── Divider ── */
    .custom-divider {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(99, 102, 241, 0.3), transparent);
        margin: 1rem 0;
    }

    /* ── Hide default streamlit elements ── */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ─────────────────────── Helper Functions ───────────────────────
def check_health():
    """Check backend health."""
    try:
        resp = requests.get(f"{API_URL}/health", timeout=5)
        return resp.json()
    except Exception:
        return None


def upload_document(file):
    """Upload a PDF to the backend."""
    try:
        files = {"file": (file.name, file.getvalue(), "application/pdf")}
        resp = requests.post(f"{API_URL}/upload", files=files, timeout=120)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.HTTPError as e:
        error_detail = e.response.json().get("detail", str(e)) if e.response is not None else str(e)
        return {"error": error_detail}
    except Exception as e:
        return {"error": str(e)}


def query_rag(question, top_k=5):
    """Send a query to the RAG backend."""
    try:
        resp = requests.post(
            f"{API_URL}/query",
            json={"question": question, "top_k": top_k},
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.HTTPError as e:
        error_detail = e.response.json().get("detail", str(e)) if e.response is not None else str(e)
        return {"error": error_detail}
    except Exception as e:
        return {"error": str(e)}


def get_documents():
    """Fetch list of indexed documents."""
    try:
        resp = requests.get(f"{API_URL}/documents", timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return []


def delete_document(doc_id):
    """Delete a document from the index."""
    try:
        resp = requests.delete(f"{API_URL}/documents/{doc_id}", timeout=30)
        resp.raise_for_status()
        return True
    except Exception:
        return False


# ─────────────────────── Session State ───────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "upload_status" not in st.session_state:
    st.session_state.upload_status = None


# ─────────────────────── Sidebar ───────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-title">🧠 RAG System</div>', unsafe_allow_html=True)

    # Health status
    health = check_health()
    if health and health.get("qdrant_connected"):
        st.markdown(
            '<span class="status-pill status-online">● System Online</span>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<span class="status-pill status-offline">● System Offline</span>',
            unsafe_allow_html=True,
        )
        if not health:
            st.caption("⚠️ Backend is not reachable. Start the FastAPI server.")
        elif not health.get("qdrant_connected"):
            st.caption("⚠️ Qdrant is not connected. Run `docker compose up -d`.")

    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

    # ── Upload Section ──
    st.markdown('<div class="sidebar-title">📄 Upload Documents</div>', unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "Drop PDFs here",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files:
        for uploaded_file in uploaded_files:
            # Use a key based on file name to avoid re-uploading same file
            file_key = f"uploaded_{uploaded_file.name}_{uploaded_file.size}"
            if file_key not in st.session_state:
                with st.spinner(f"📤 Indexing `{uploaded_file.name}`..."):
                    result = upload_document(uploaded_file)
                    if "error" in result:
                        st.error(f"❌ {result['error']}")
                    else:
                        st.success(
                            f"✅ **{result['filename']}**\n\n"
                            f"Indexed **{result['num_chunks']}** chunks"
                        )
                        st.session_state[file_key] = True
                        time.sleep(0.5)
                        st.rerun()

    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

    # ── Document List ──
    st.markdown('<div class="sidebar-title">📚 Indexed Documents</div>', unsafe_allow_html=True)

    documents = get_documents()
    if documents:
        for doc in documents:
            col1, col2 = st.columns([4, 1])
            with col1:
                st.markdown(
                    f'<div class="doc-card">'
                    f'<div class="doc-name">📄 {doc["filename"]}</div>'
                    f'<div class="doc-meta">{doc["num_chunks"]} chunks</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            with col2:
                if st.button("🗑️", key=f"del_{doc['document_id']}", help=f"Delete {doc['filename']}"):
                    with st.spinner("Deleting..."):
                        if delete_document(doc["document_id"]):
                            st.rerun()
                        else:
                            st.error("Failed to delete")
    else:
        st.caption("No documents indexed yet. Upload a PDF to get started.")

    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

    # ── Settings ──
    st.markdown('<div class="sidebar-title">⚙️ Settings</div>', unsafe_allow_html=True)
    top_k = st.slider("Retrieval chunks (top_k)", min_value=1, max_value=20, value=5)

    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()


# ─────────────────────── Main Content ───────────────────────
st.markdown(
    '<div class="hero-header">'
    "<h1>🧠 Intelligent Document Q&A</h1>"
    "<p>Upload PDFs and ask questions — powered by Groq & Qdrant</p>"
    "</div>",
    unsafe_allow_html=True,
)

st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

# ── Chat History ──
for entry in st.session_state.chat_history:
    if entry["role"] == "user":
        st.markdown(
            f'<div class="chat-message user-msg">💬 &nbsp; {entry["content"]}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="chat-message bot-msg">🤖 &nbsp; {entry["content"]}</div>',
            unsafe_allow_html=True,
        )
        # Source chips
        if entry.get("sources"):
            source_html = " ".join(
                f'<span class="source-chip">📄 {s["document_name"]} (chunk {s["chunk_index"]}, '
                f'score: {s["score"]:.2f})</span>'
                for s in entry["sources"]
            )
            st.markdown(source_html, unsafe_allow_html=True)

# ── Chat Input ──
if prompt := st.chat_input("Ask a question about your documents..."):
    # Add user message
    st.session_state.chat_history.append({"role": "user", "content": prompt})

    # Display user message
    st.markdown(
        f'<div class="chat-message user-msg">💬 &nbsp; {prompt}</div>',
        unsafe_allow_html=True,
    )

    # Query backend
    with st.spinner("🔍 Searching & generating answer..."):
        result = query_rag(prompt, top_k=top_k)

    if "error" in result:
        error_msg = f"⚠️ Error: {result['error']}"
        st.session_state.chat_history.append({"role": "assistant", "content": error_msg, "sources": []})
        st.markdown(
            f'<div class="chat-message bot-msg">🤖 &nbsp; {error_msg}</div>',
            unsafe_allow_html=True,
        )
    else:
        answer = result.get("answer", "No answer generated.")
        sources = result.get("sources", [])

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer,
            "sources": sources,
        })

        st.markdown(
            f'<div class="chat-message bot-msg">🤖 &nbsp; {answer}</div>',
            unsafe_allow_html=True,
        )

        if sources:
            source_html = " ".join(
                f'<span class="source-chip">📄 {s["document_name"]} (chunk {s["chunk_index"]}, '
                f'score: {s["score"]:.2f})</span>'
                for s in sources
            )
            st.markdown(source_html, unsafe_allow_html=True)

    st.rerun()
