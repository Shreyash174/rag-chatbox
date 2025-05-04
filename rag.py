import os
import hashlib
import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings  # ✅ Updated import
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
import warnings
import argparse
import pickle


warnings.filterwarnings("ignore", category=DeprecationWarning, module="langchain.*")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  

CACHE_DIR = "vector_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def scrape_website(url: str) -> str:
    """Scrapes and cleans a website's text content."""
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        for tag in soup(['script', 'style', 'nav', 'footer', 'iframe', 'noscript']):
            tag.decompose()

        return ' '.join(soup.stripped_strings)
    except Exception as e:
        print(f"❌ Scraping failed: {e}")
        return ""

def split_text(text: str) -> list:
    """Splits large text into overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len
    )
    return splitter.split_text(text)

def get_cache_path(content: str) -> str:
    """Generates a cache filename based on content hash."""
    content_hash = hashlib.md5(content.encode()).hexdigest()
    return os.path.join(CACHE_DIR, f"{content_hash}.pkl")

def create_or_load_vectorstore(chunks: list, cache_path: str):
    """Creates or loads vectorstore from cache."""
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            print("✅ Loaded vectorstore from cache.")
            return pickle.load(f)

    print("📦 Generating new vectorstore...")
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    vectorstore = FAISS.from_texts(chunks, embeddings)

    with open(cache_path, "wb") as f:
        pickle.dump(vectorstore, f)

    return vectorstore

def create_qa_chain(vectorstore):
    """Initializes the RetrievalQA chain with local LLM."""
    try:
        llm = Ollama(
            model="tinyllama",
            temperature=0.3,
            num_ctx=2048
        )
        return RetrievalQA.from_chain_type(
            llm=llm,
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
            chain_type="stuff"
        )
    except Exception as e:
        print(f"⚠️ LLM initialization failed: {e}")
        return vectorstore.as_retriever()

def run_rag(url: str, question: str):
    print("🕸️ Scraping website...")
    content = scrape_website(url)
    if not content:
        print("⚠️ No content retrieved. Exiting.")
        return

    print("✂️ Splitting text...")
    chunks = split_text(content)

    print("📁 Handling vectorstore cache...")
    cache_path = get_cache_path(content)
    vectorstore = create_or_load_vectorstore(chunks, cache_path)

    print("🔍 Preparing QA chain...")
    qa_chain = create_qa_chain(vectorstore)

    print("\n💬 Answering your question...")
    try:
        if hasattr(qa_chain, 'invoke'):
            result = qa_chain.invoke({"query": question})
            print("🧠 Answer:", result["result"])
        else:
            docs = qa_chain.get_relevant_documents(question)
            print("🔎 Top relevant content snippet:\n")
            print(docs[0].page_content[:1000] + "...")
    except Exception as e:
        print(f"❌ Failed to generate answer: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG-Powered Website Chatbot")
    parser.add_argument("--url", help="Website URL", required=False)
    parser.add_argument("--question", help="Your question", required=False)
    args = parser.parse_args()

    if args.url and args.question:
        run_rag(args.url, args.question)
    else:
        print("🤖 RAG-Powered Website Chatbot (CLI or Interactive Mode)")
        url = input("🔗 Enter website URL: ").strip()
        question = input("❓ Enter your question: ").strip()
        run_rag(url, question)
