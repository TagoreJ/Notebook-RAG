import streamlit as st
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader

# ===========================
# 1️⃣ Streamlit Setup
# ===========================
st.set_page_config(page_title="Notebook RAG", layout="wide")
st.title("📘 Notebook RAG (LangChain + Chroma + Gemini)")

# ===========================
# 2️⃣ API Key Setup
# ===========================
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY

# ===========================
# 3️⃣ File Upload
# ===========================
uploaded_files = st.file_uploader(
    "📂 Upload your notes (PDF, TXT, DOCX)",
    type=["pdf", "txt", "docx"],
    accept_multiple_files=True
)

if not uploaded_files:
    st.info("Please upload at least one file to start.")
    st.stop()

# ===========================
# 4️⃣ Load Documents
# ===========================
docs = []
for file in uploaded_files:
    ext = file.name.split(".")[-1].lower()
    path = f"./temp_{file.name}"
    with open(path, "wb") as f:
        f.write(file.read())

    if ext == "pdf":
        loader = PyPDFLoader(path)
    elif ext == "txt":
        loader = TextLoader(path)
    elif ext == "docx":
        loader = Docx2txtLoader(path)
    else:
        st.warning(f"❌ Unsupported file: {file.name}")
        continue
    docs.extend(loader.load())

# ===========================
# 5️⃣ Text Splitter
# ===========================
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
split_docs = splitter.split_documents(docs)

# ===========================
# 6️⃣ Embeddings + Vector DB
# ===========================
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vectordb = Chroma.from_documents(split_docs, embeddings, persist_directory="chroma_store")
vectordb.persist()
retriever = vectordb.as_retriever(search_kwargs={"k": 3})

# ===========================
# 7️⃣ LLM + QA Chain
# ===========================
llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.2)
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")

# ===========================
# 8️⃣ Chat Interface
# ===========================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

query = st.text_input("💬 Ask a question from your uploaded files:")

if query:
    with st.spinner("Thinking..."):
        answer = qa_chain.run(query)
        st.session_state.chat_history.append((query, answer))

# ===========================
# 9️⃣ Display Conversation
# ===========================
if st.session_state.chat_history:
    st.subheader("🧾 Chat History")
    for i, (q, a) in enumerate(st.session_state.chat_history[::-1]):
        st.markdown(f"**Q{i+1}:** {q}")
        st.markdown(f"**A{i+1}:** {a}")
        st.markdown("---")
