import os
import streamlit as st
import pdfplumber
import docx
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import LlamaCpp
from langchain.chains import RetrievalQA

# Load the local LLM model only once
@st.cache_resource
def load_llm():
    return LlamaCpp(
        model_path="./mistral.gguf",  # Local GGUF model path
        n_ctx=2048,
        temperature=0.7,
        max_tokens=512,
        top_p=0.95,
        n_batch=16,
        verbose=False,
    )

# Load text from different document types
def load_pdf(file):
    with pdfplumber.open(file) as pdf:
        return "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])

def load_docx(file):
    doc = docx.Document(file)
    return "\n".join([para.text for para in doc.paragraphs if para.text.strip()])

def load_text(file):
    return file.read().decode("utf-8")

def load_file(uploaded_file):
    name = uploaded_file.name.lower()
    if name.endswith(".pdf"):
        return load_pdf(uploaded_file)
    elif name.endswith(".docx"):
        return load_docx(uploaded_file)
    elif name.endswith(".txt"):
        return load_text(uploaded_file)
    else:
        return None

# QA System
def create_qa_chain(text, llm):
    if not text.strip():
        raise ValueError("The document appears to be empty.")

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.create_documents([text])
    if not chunks:
        raise ValueError("Text splitting produced no usable chunks.")

    # ✅ USE LOCAL embedding model (must already be downloaded)
    embeddings = HuggingFaceEmbeddings(model_name="./models/all-MiniLM-L6-v2")

    try:
        vectorstore = FAISS.from_documents(chunks, embeddings)
    except IndexError:
        raise ValueError("Embedding failed. Check text and model.")

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return qa

# Streamlit UI
st.set_page_config(page_title="Offline Document Q&A", layout="centered")
st.title(" Offline Document Q&A Assistant")

uploaded_file = st.file_uploader("Upload a document (.pdf, .docx, .txt)", type=["pdf", "docx", "txt"])

if uploaded_file:
    st.success("✅ Document uploaded successfully!")
    with st.spinner("⏳ Processing document..."):
        try:
            text = load_file(uploaded_file)
            llm = load_llm()
            qa_chain = create_qa_chain(text, llm)
            st.success(" Ready! Ask your question below.")
        except Exception as e:
            st.error(f"Error processing document: {e}")
            qa_chain = None

    if qa_chain:
        query = st.text_input("Ask a question about the document:")
        if query:
            with st.spinner("🤔"):
                try:
                    answer = qa_chain.run(query)
                    st.markdown(f"**Answer:** {answer}")
                except Exception as e:
                    st.error(f"Failed to generate answer: {e}")
