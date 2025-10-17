import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.document_loaders import TextLoader
import os

# ===========================
# 1️⃣ Streamlit Setup
# ===========================
st.set_page_config(page_title="Notebook RAG", layout="wide")
st.title("📒 Notebook RAG using LangChain + ChromaDB")

# ===========================
# 2️⃣ API Key Setup
# ===========================
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY

# ===========================
# 3️⃣ File Upload
# ===========================
uploaded_file = st.file_uploader("📂 Upload your text or notes file (.txt)", type=["txt"])
if uploaded_file:
    content = uploaded_file.read().decode("utf-8")
    st.success("✅ File uploaded successfully!")
else:
    st.info("Please upload a text file to begin.")
    st.stop()

# ===========================
# 4️⃣ Text Splitter
# ===========================
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
texts = text_splitter.create_documents([content])

# ===========================
# 5️⃣ Embeddings & Vector Store
# ===========================
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Store locally in Chroma
persist_directory = "chroma_store"
vectordb = Chroma.from_documents(documents=texts, embedding=embeddings, persist_directory=persist_directory)
vectordb.persist()

retriever = vectordb.as_retriever(search_kwargs={"k": 3})

# ===========================
# 6️⃣ LLM Setup (Gemini)
# ===========================
llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.2)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff"
)

# ===========================
# 7️⃣ Chat Interface
# ===========================
query = st.text_input("💬 Ask a question based on your notes:")
if query:
    with st.spinner("Thinking..."):
        response = qa_chain.run(query)
        st.markdown(f"**Answer:** {response}")
