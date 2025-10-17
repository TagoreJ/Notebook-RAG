import os
import uuid
import hashlib
import streamlit as st
from typing import List
# Changed from PyPDF2 to pypdf as PyPDF2 is deprecated
from pypdf import PdfReader
import docx
from tqdm import tqdm
# Renamed to avoid conflict with the library name
import google.generativeai as genai
from pinecone import Pinecone

# ============================
# 1️⃣ Load Secrets
# ============================
try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
except Exception:
    st.error("⚠️ Please configure your API keys in Streamlit Cloud (Settings → Secrets).")
    st.stop()

# ============================
# 2️⃣ Initialize Clients
# ============================
# Google Gemini
genai.configure(api_key=GEMINI_API_KEY)

# Pinecone client
pinecone_client = Pinecone(api_key=PINECONE_API_KEY)

# ============================
# 3️⃣ Fixed Pinecone Index
# ============================
INDEX_NAME = "sainotes"
# Check if the index exists, otherwise, ask the user to create it
if INDEX_NAME not in pinecone_client.list_indexes().names():
    st.error(f"🚨 Pinecone index '{INDEX_NAME}' not found. Please create it in your Pinecone dashboard.")
    st.stop()
index = pinecone_client.Index(INDEX_NAME)

# ============================
# 4️⃣ Multi-user Namespace
# ============================
if "namespace" not in st.session_state:
    st.session_state["namespace"] = str(uuid.uuid4())
SESSION_NAMESPACE = st.session_state["namespace"]

# ============================
# 5️⃣ Helper Functions
# ============================
def extract_text_from_pdf(file):
    # Using pypdf's PdfReader
    reader = PdfReader(file)
    return "\n".join(page.extract_text() or "" for page in reader.pages)

def extract_text_from_docx(file):
    doc = docx.Document(file)
    return "\n".join(p.text for p in doc.paragraphs)

def extract_text_from_txt(file):
    return file.read().decode("utf-8")

def extract_text(file):
    name = file.name.lower()
    if name.endswith(".pdf"):
        return extract_text_from_pdf(file)
    elif name.endswith(".docx") or name.endswith(".doc"):
        return extract_text_from_docx(file)
    elif name.endswith(".txt"):
        return extract_text_from_txt(file)
    else:
        try:
            return file.read().decode("utf-8")
        except Exception:
            return ""

def chunk_text(text, chunk_size=1000, overlap=150):
    tokens = text.split()
    chunks, i = [], 0
    while i < len(tokens):
        chunks.append(" ".join(tokens[i:i + chunk_size]))
        i += chunk_size - overlap
    return [chunk for chunk in chunks if chunk] # Ensure no empty chunks

def embed_texts(texts, model="models/embedding-001"):
    # Corrected model name
    try:
        # Note: The 'contents' parameter expects a list of strings
        res = genai.embed_content(model=model, content=texts, task_type="retrieval_document")
        return res['embedding']
    except Exception as e:
        st.error(f"Error embedding texts: {e}")
        return []

def upsert_chunks(chunks, meta_base, namespace):
    BATCH_SIZE = 50
    for i in tqdm(range(0, len(chunks), BATCH_SIZE), desc="Embedding & Upserting"):
        batch_texts = chunks[i:i + BATCH_SIZE]
        embeddings = embed_texts(batch_texts)

        if not embeddings: # Skip if embedding failed
            continue

        vectors = []
        # The length of embeddings should match batch_texts
        for j, emb in enumerate(embeddings):
            meta = meta_base.copy()
            meta.update({
                "chunk_index": i + j,
                "text_preview": batch_texts[j][:300],
            })
            # ✅ FIX: Pass the numerical vector using emb, not the whole object
            vectors.append((str(uuid.uuid4()), emb, meta))
        
        if vectors:
            index.upsert(vectors=vectors, namespace=namespace)

def query_pinecone(query, top_k=4, model="models/embedding-001", namespace="default"):
    # ✅ FIX: Use genai.embed_content for the query and access the embedding list
    try:
        query_embedding = genai.embed_content(
            model=model,
            content=query,
            task_type="retrieval_query"
        )['embedding']
        return index.query(vector=query_embedding, top_k=top_k, include_metadata=True, namespace=namespace)
    except Exception as e:
        st.error(f"Error querying Pinecone: {e}")
        return None

def build_prompt(chunks, question):
    ctx = ""
    for m in chunks:
        md = m.get("metadata", {})
        ctx += f"\n---\nSource: {md.get('source','?')} | Chunk: {md.get('chunk_index','?')}\n{md.get('text_preview','')}"
    return f"""
You are a helpful assistant. Use ONLY the below context to answer the user's question.
If the answer isn't in the context, reply with "I don't know."

Context:
{ctx}

Question: {question}

Answer:
"""

def generate_answer(prompt, model="gemini-1.5-flash"):
    # Corrected model name
    model = genai.GenerativeModel(model)
    return model.generate_content(prompt).text

def delete_namespace(namespace):
    try:
        index.delete(delete_all=True, namespace=namespace)
        return True
    except Exception as e:
        st.error(f"Error deleting namespace: {e}")
        return False

# ============================
# 6️⃣ Streamlit UI
# ============================
st.set_page_config(page_title="Google RAG + Pinecone", layout="wide")
st.title("📚 Streamlit RAG with Google Gemini + Pinecone")
st.caption("Each user session has its own namespace for multi-user isolation.")

with st.sidebar:
    st.header("🔒 API Configuration")
    st.success("API keys loaded securely from Streamlit Secrets.")
    st.write(f"**Pinecone Index:** {INDEX_NAME}")
    st.write(f"**Namespace:** `{SESSION_NAMESPACE}`")
    st.markdown("---")

# ---------- File Upload ----------
uploaded_file = st.file_uploader("📁 Upload file (PDF/DOCX/TXT):", type=["pdf", "docx", "txt"])

if uploaded_file:
    file_size = uploaded_file.size
    doc_id = hashlib.sha1(f"{uploaded_file.name}_{file_size}".encode()).hexdigest()
    st.info(f"File uploaded: **{uploaded_file.name}** | Size: {round(file_size/1024,1)} KB")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("📤 Index File"):
            with st.spinner("Extracting, chunking, and embedding file..."):
                text = extract_text(uploaded_file)
                if not text or not text.strip():
                    st.error("Could not extract text from file or file is empty.")
                else:
                    chunks = chunk_text(text, chunk_size=800, overlap=150)
                    meta_base = {"source": uploaded_file.name, "doc_id": doc_id}
                    upsert_chunks(chunks, meta_base, namespace=SESSION_NAMESPACE)
                    st.success("✅ File indexed successfully.")

    with col2:
        if st.button("🗑️ Delete Session Data"):
            if delete_namespace(SESSION_NAMESPACE):
                st.success("✅ Deleted all vectors for this session.")

st.divider()

# ---------- Q&A Section ----------
st.header("💬 Ask your question")
question = st.text_input("Enter your question:", key="question_input")
top_k = st.slider("Top K results to fetch", 1, 8, 4)

if st.button("🔍 Get Answer") and question.strip():
    with st.spinner("Retrieving from Pinecone..."):
        results = query_pinecone(question, top_k, namespace=SESSION_NAMESPACE)

    if not results or not results.matches:
        st.warning("No matching results found in Pinecone.")
    else:
        st.write("### 🔹 Top Retrieved Chunks")
        for m in results.matches:
            md = m.metadata
            st.markdown(f"- **{md.get('source')}** | Score: {m.score:.4f}")
            with st.expander("Show Text"):
                 st.caption(md.get("text_preview",""))

        prompt = build_prompt(results.matches, question)
        with st.spinner("Generating answer with Gemini..."):
            answer = generate_answer(prompt)
        st.subheader("🧠 Answer")
        st.write(answer)

st.markdown("---")
st.caption("Built with ❤️ using Google Gemini + Pinecone + Streamlit")

# ============================
# 7️⃣ Cleanup: Removed st.on_session_end
# ============================
# ✅ FIX: The st.on_session_end function does not exist and was removed.
# Users can use the "Delete Session Data" button for cleanup.