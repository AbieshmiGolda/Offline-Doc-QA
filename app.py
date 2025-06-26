# app.py

import os
import streamlit as st
import pdfplumber
import docx
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import LlamaCpp
from langchain.chains import RetrievalQA

# === STEP 1: Load the local LLM model only once ===
@st.cache_resource
def load_llm():
    return LlamaCpp(
        model_path="./mistral.gguf",  # Path to your downloaded GGUF file
        n_ctx=2048,
        temperature=0.7,
        max_tokens=512,
        top_p=0.95,
        n_batch=16,
        verbose=False,
    )

# === STEP 2: Load text from document ===
def load_pdf(file):
    with pdfplumber.open(file) as pdf:
        return "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])

def load_docx(file):
    doc = docx.Document(file)
    return "\n".join([para.text for para in doc.paragraphs])

def load_text(file):
    return file.read().decode('utf-8')

def load_file(uploaded_file):
    name = uploaded_file.name.lower()
    if name.endswith('.pdf'):
        return load_pdf(uploaded_file)
    elif name.endswith('.docx'):
        return load_docx(uploaded_file)
    elif name.endswith('.txt'):
        return load_text(uploaded_file)
    else:
        return None

# === STEP 3: Create QA system using vector store + local LLM ===
def create_qa_chain(text, llm):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.create_documents([text])

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return qa

# === STEP 4: Streamlit UI ===
st.set_page_config(page_title="Offline Doc Q&A", layout="centered")
st.title("📄 Offline Document Q&A Assistant")

uploaded_file = st.file_uploader("Upload a document (.pdf, .docx, .txt)", type=["pdf", "docx", "txt"])

if uploaded_file:
    st.success("✅ Document uploaded successfully!")
    with st.spinner("⏳ Reading and processing the document..."):
        text = load_file(uploaded_file)
        llm = load_llm()
        qa_chain = create_qa_chain(text, llm)
    st.success("✅ Ready! Ask your question below.")

    query = st.text_input("Ask a question about the document:")
    if query and qa_chain:
        with st.spinner("🤔 Thinking..."):
            answer = qa_chain.run(query)
        st.markdown(f"**Answer:** {answer}")
