import os
import uuid
import hashlib
import time
import pandas as pd
import streamlit as st
from typing import List
from pathlib import Path

# File processing
from pypdf import PdfReader
import docx
import nbformat
import json

# AI and Vector DB
import google.generativeai as genai
from pinecone import Pinecone

# ============================
# 1️⃣ Load Secrets & Configure
# ============================
try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
except KeyError:
    st.error("⚠️ Please configure GEMINI_API_KEY and PINECONE_API_KEY in Streamlit's Secrets.")
    st.stop()

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)

# Configure Pinecone
INDEX_NAME = "sainotes" # Make sure this index exists
pinecone_client = Pinecone(api_key=PINECONE_API_KEY)
if INDEX_NAME not in pinecone_client.list_indexes().names():
    st.error(f"🚨 Pinecone index '{INDEX_NAME}' not found. Please create it first.")
    st.stop()
index = pinecone_client.Index(INDEX_NAME)

# ============================
# 2️⃣ Session State and Namespace
# ============================
if "namespace" not in st.session_state:
    st.session_state.namespace = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
SESSION_NAMESPACE = st.session_state.namespace

# ============================
# 3️⃣ ✨ NEW: Resilient Gemini API Call Function
# ============================
def gemini_api_call_with_retry(api_call_func, retries=3, **kwargs):
    """
    Wrapper to make Gemini API calls more resilient to rate-limiting (429) errors.
    """
    for attempt in range(retries):
        try:
            return api_call_func(**kwargs)
        except Exception as e:
            if "429" in str(e):
                wait_time = min(60, (attempt + 1) * 15) # Exponential backoff
                st.warning(f"⚠️ Rate limit hit. Retrying in {wait_time} seconds... (Attempt {attempt+1}/{retries})")
                time.sleep(wait_time)
                continue
            else:
                st.error(f"An unexpected error occurred: {e}")
                return None
    st.error("API call failed after multiple retries. Please try again later.")
    return None

# ============================
# 4️⃣ Helper Functions (File Processing, RAG Pipeline)
# ============================

# --- File Extractors ---
def extract_text(uploaded_file):
    """Extracts text from various file formats."""
    name = uploaded_file.name.lower()
    # Read file into a temporary location to ensure consistent handling
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(name).suffix) as tmp:
        tmp.write(uploaded_file.getvalue())
        path = tmp.name

    content = ""
    try:
        if name.endswith(".pdf"):
            reader = PdfReader(path)
            content = "\n".join(page.extract_text() or "" for page in reader.pages)
        elif name.endswith((".docx", ".doc")):
            doc = docx.Document(path)
            content = "\n".join(p.text for p in doc.paragraphs)
        elif name.endswith(".ipynb"):
            nb = nbformat.read(path, as_version=4)
            for cell in nb.cells:
                if cell.cell_type == "markdown":
                    content += f"## Markdown Cell\n{cell.source}\n\n"
                elif cell.cell_type == "code":
                    content += f"## Code Cell\n```python\n{cell.source}\n```\n\n"
        else: # .txt, .py, .md, etc.
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
    except Exception as e:
        st.error(f"Error processing file {name}: {e}")
    finally:
        os.unlink(path) # Clean up temp file
    return content

# --- RAG Core Functions ---
def chunk_text(text, chunk_size=800, overlap=100):
    tokens = text.split()
    return [" ".join(tokens[i:i + chunk_size]) for i in range(0, len(tokens), chunk_size - overlap) if tokens[i:i + chunk_size]]

def embed_texts(texts, model="models/embedding-001"):
    response = gemini_api_call_with_retry(
        genai.embed_content,
        model=model,
        content=texts,
        task_type="retrieval_document"
    )
    return response['embedding'] if response else []

def upsert_chunks(chunks, meta_base, namespace, progress_bar):
    BATCH_SIZE = 100
    total_upserted = 0
    for i in range(0, len(chunks), BATCH_SIZE):
        batch_texts = chunks[i:i + BATCH_SIZE]
        embeddings = embed_texts(batch_texts)
        if not embeddings: continue

        vectors = [
            (str(uuid.uuid4()), emb, {**meta_base, "chunk_index": i + j, "text_preview": batch_texts[j][:400]})
            for j, emb in enumerate(embeddings)
        ]
        if vectors:
            index.upsert(vectors=vectors, namespace=namespace)
            total_upserted += len(vectors)
        progress_bar.progress(min((i + BATCH_SIZE) / len(chunks), 1.0), text=f"Upserting batch {i//BATCH_SIZE + 1}...")
    return total_upserted

def query_pinecone(query, top_k=4, model="models/embedding-001", namespace="default"):
    embedding_response = gemini_api_call_with_retry(
        genai.embed_content,
        model=model,
        content=query,
        task_type="retrieval_query"
    )
    if not embedding_response: return None
    
    query_embedding = embedding_response['embedding']
    return index.query(vector=query_embedding, top_k=top_k, include_metadata=True, namespace=namespace)

def build_prompt(chunks, question):
    context = "\n---\n".join([f"Source: {m.metadata.get('source', '?')}\nContent: {m.metadata.get('text_preview', '')}" for m in chunks])
    return f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"

def generate_answer(prompt, model_name="gemini-1.5-flash"):
    model = genai.GenerativeModel(model_name)
    response = gemini_api_call_with_retry(
        model.generate_content,
        contents=prompt
    )
    return response.text if response else "Sorry, I couldn't generate an answer due to an API issue."

def delete_namespace(namespace):
    try:
        index.delete(delete_all=True, namespace=namespace)
        return True
    except Exception as e:
        st.error(f"Error deleting namespace: {e}")
        return False

# ============================
# 5️⃣ Streamlit UI
# ============================
st.set_page_config(page_title="Resilient RAG", page_icon="🛡️", layout="wide")
st.title("🛡️ Resilient RAG with Gemini, Pinecone & Rate-Limit Handling")

with st.sidebar:
    st.header("⚙️ Configuration")
    st.success("API keys loaded successfully.")
    st.info(f"**Pinecone Index:** `{INDEX_NAME}`")
    st.warning(f"**Session Namespace:** `{SESSION_NAMESPACE}`")
    if st.button("🗑️ Delete Session Data", use_container_width=True):
        delete_namespace(SESSION_NAMESPACE)
        st.session_state.messages = []
        st.success("✅ Session data and chat history cleared.")
        st.rerun()

# --- Part 1: File Upload and Indexing ---
st.header("1. Index Your Document")
file_types = ["pdf", "docx", "txt", "md", "py", "js", "java", "cpp", "json", "ipynb"]
uploaded_file = st.file_uploader("📁 Choose a file", type=file_types)

if uploaded_file:
    if st.button(f"🚀 Index '{uploaded_file.name}'"):
        with st.status("Processing file...", expanded=True) as status:
            text = extract_text(uploaded_file)
            if text and text.strip():
                st.write(f"✅ Text extracted ({len(text)} chars). Chunking now...")
                chunks = chunk_text(text)
                st.write(f"✅ Divided into {len(chunks)} chunks. Upserting to Pinecone...")
                progress_bar = st.progress(0, text="Starting upsert...")
                total_upserted = upsert_chunks(chunks, {"source": uploaded_file.name}, SESSION_NAMESPACE, progress_bar)
                status.update(label="✅ Indexing Complete!", state="complete")

                st.subheader("📊 Indexing Report")
                st.metric(label="Vectors Upserted to Pinecone", value=total_upserted)
            else:
                status.update(label="❌ Error: Could not extract text.", state="error")

st.divider()

# --- Part 2: Chat Interface ---
st.header("2. Ask Questions")

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask a question about your document..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            results = query_pinecone(prompt, top_k=5, namespace=SESSION_NAMESPACE)
            
            if not results or not results.matches:
                st.warning("Could not find relevant context in your document to answer this question.")
            else:
                # Display retrieved context in an expander
                with st.expander("📚 View Retrieved Context"):
                    retrieved_data = [{"Score": m.score, "Text": m.metadata.get('text_preview', '')} for m in results.matches]
                    st.dataframe(pd.DataFrame(retrieved_data), use_container_width=True)

                final_prompt = build_prompt(results.matches, prompt)
                answer = generate_answer(final_prompt)
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})