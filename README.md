
# 🤖 RAG-Powered Website Chatbot

A web + CLI tool built with LangChain, FAISS, and Ollama that lets you ask questions about any webpage using Retrieval-Augmented Generation (RAG).

## 🔧 Features
- Scrapes any URL
- Splits text into chunks
- Embeds using `sentence-transformers`
- Stores/retrieves using FAISS
- Uses `Ollama` + `tinyllama` for local LLM inference
- Supports both Streamlit UI and CLI mode

## 🚀 Usage

### 🖥️ CLI
```bash
python app.py --url https://example.com --question "What is this page about?"

