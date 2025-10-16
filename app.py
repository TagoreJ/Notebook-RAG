import os
import uuid
import hashlib
import streamlit as st
from typing import List
import PyPDF2
import docx
from tqdm import tqdm
from google import genai
from pinecone import Pinecone
from pinecone.models import ServerlessSpec

# ============================
# 1️⃣ Load Secrets
# ============================
try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
    PINECONE_INDEX = st.secrets["PINECONE_INDEX"]
    PINECONE_REGION = st.secrets.get("PINECONE_REGION", "us-east-1")  # default region
except Exception:
    st.error("⚠️ Please configure your API keys in Streamlit Cloud (Settings → Secrets).")
    st.stop()

# ============================
# 2️⃣ Initialize Clients
# ============================
client = genai.Client(api_key=GEMINI_API_KEY)
pinecone_client = Pinecone(api_key=PINECONE_API_KEY)

# ============================
# 3️⃣ Create / Connect Serverless Index
# ============================
spec = ServerlessSpec(cloud="aws", region=PINECONE_REGION)

if PINECONE_INDEX not in pinecone_client.list_indexes():
    pinecone_client.create_index(
        name=PINECONE_INDEX,
        dimension=768,   # Gemini embedding dimension
        metric="cosine",
        spec=spec
    )

index = pinecone_client.Index(PINECONE_INDEX)

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
    reader = PyPDF2.PdfReader(file)
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
    return chunks

def embed_texts(texts, model="gemini-embedding-001"):
    res = client.models.embed_content(model=model, contents=texts)
    return res.embeddings

def compute_doc_id(filename, size):
    return hashlib.sha1(f"{filename}_{size}".encode("utf-8")).hexdigest()

def upsert_chunks(chunks, meta_base, namespace):
    BATCH_SIZE = 50
    for i in tqdm(range(0, len(chunks), BATCH_SIZE), desc="Embedding"):
        batch_texts = chunks[i:i + BATCH_SIZE]
        embeddings = embed_texts(batch_texts)
        vectors = []
        for j, emb in enumerate(embeddings):
            meta = meta_base.copy()
            meta.update({
                "chunk_index": i + j,
                "text_preview": batch_texts[j][:300],
            })
            vectors.append((str(uuid.uuid4()), emb, meta))
        index.upsert(vectors=vectors, namespace=namespace)

def query_pinecone(query, top_k=4, model="gemini-embedding-001", namespace="default"):
    emb = client.models.embed_content(model=model, contents=query).embeddings[0]
    return index.query(vector=emb, top_k=top_k, include_metadata=True, namespace=namespace)

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

def generate_answer(prompt, model="gemini-2.5-flash"):
    return client.models.generate_content(model=model, contents=prompt).text

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
    st.success("All API keys loaded securely from Streamlit Secrets.")
    st.write(f"**Pinecone Index:** {PINECONE_INDEX}")
    st.write(f"**Namespace:** {SESSION_NAMESPACE}")
    st.markdown("---")

uploaded_file = st.file_uploader("📁 Upload file (PDF/DOCX/TXT):", type=["pdf", "docx", "txt"])

if uploaded_file:
    file_size = uploaded_file.size
    doc_id = compute_doc_id(uploaded_file.name, file_size)
    st.info(f"File uploaded: **{uploaded_file.name}** | Size: {round(file_size/1024,1)} KB")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("📤 Index File"):
            with st.spinner("Extracting and embedding file..."):
                text = extract_text(uploaded_file)
                if not text.strip():
                    st.error("Could not extract text from file.")
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

st.header("💬 Ask your question")
question = st.text_input("Enter your question:")
top_k = st.slider("Top K results to fetch", 1, 8, 4)

if st.button("🔍 Get Answer") and question.strip():
    with st.spinner("Retrieving from Pinecone..."):
        results = query_pinecone(question, top_k, namespace=SESSION_NAMESPACE)

    matches = getattr(results, "matches", [])
    if not matches:
        st.warning("No matching results found.")
    else:
        st.write("### 🔹 Top Retrieved Chunks")
        for m in matches:
            md = m.metadata
            st.markdown(f"- **{md.get('source')}** | Score: {m.score:.4f}")
            st.caption(md.get("text_preview","")[:300])

        prompt = build_prompt(matches, question)
        with st.spinner("Generating answer with Gemini..."):
            answer = generate_answer(prompt)
        st.subheader("🧠 Answer")
        st.write(answer)

st.markdown("---")
st.caption("Built with ❤️ using Google Gemini + Pinecone + Streamlit")

# ============================
# 7️⃣ Cleanup: Delete namespace on session end
# ============================
def cleanup():
    delete_namespace(SESSION_NAMESPACE)

st.on_session_end(cleanup)
