import os
import hashlib
import pickle
import requests
from bs4 import BeautifulSoup
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
import warnings

# Suppress warnings and logs
warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

CACHE_DIR = "vector_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def scrape_website(url: str) -> str:
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        for tag in soup(['script', 'style', 'nav', 'footer', 'iframe', 'noscript']):
            tag.decompose()
        return ' '.join(soup.stripped_strings)
    except Exception as e:
        st.error(f"❌ Error scraping website: {e}")
        return ""

def split_text(text: str) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len
    )
    return splitter.split_text(text)

def get_cache_path(content: str) -> str:
    content_hash = hashlib.md5(content.encode()).hexdigest()
    return os.path.join(CACHE_DIR, f"{content_hash}.pkl")

def create_or_load_vectorstore(chunks: list, cache_path: str):
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    vectorstore = FAISS.from_texts(chunks, embeddings)

    with open(cache_path, "wb") as f:
        pickle.dump(vectorstore, f)

    return vectorstore

def create_qa_chain(vectorstore):
    try:
        llm = Ollama(model="tinyllama", temperature=0.3, num_ctx=2048)
        return RetrievalQA.from_chain_type(
            llm=llm,
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
            chain_type="stuff"
        )
    except Exception as e:
        st.warning(f"⚠️ Error setting up LLM: {e}")
        return vectorstore.as_retriever()

# UI Layout
st.set_page_config(page_title="RAG Website Chatbot", page_icon="🤖")
st.title("🤖 RAG-Powered Website Chatbot")
st.markdown("Ask questions about any webpage using Retrieval-Augmented Generation (RAG).")

url = st.text_input("🔗 Enter website URL")
question = st.text_input("❓ Enter your question")

if st.button("Ask"):
    if not url or not question:
        st.warning("Please provide both the website URL and your question.")
    else:
        with st.spinner("🕸️ Scraping and preparing data..."):
            content = scrape_website(url)
            if not content:
                st.stop()

            chunks = split_text(content)
            cache_path = get_cache_path(content)
            vectorstore = create_or_load_vectorstore(chunks, cache_path)
            qa_chain = create_qa_chain(vectorstore)

        with st.spinner("💬 Generating answer..."):
            try:
                if hasattr(qa_chain, 'invoke'):
                    result = qa_chain.invoke({"query": question})
                    st.success(result["result"])
                else:
                    docs = qa_chain.get_relevant_documents(question)
                    st.info(docs[0].page_content[:1000] + "...")
            except Exception as e:
                st.error(f"❌ Failed to generate answer: {e}")
