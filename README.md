# RAG Chatbot

An AI-powered study assistant that answers questions from your own PDF notes and previous year question papers (PYQs) using Retrieval-Augmented Generation (RAG).

Upload your notes, ask a question, and get answers grounded in your actual course material — not generic internet knowledge.

## Features

- 📄 Upload PDF notes/PYQs directly from the browser
- 🔍 Semantic search over your documents using FAISS
- 🤖 Fast, accurate answers powered by Groq's Llama 3.3 70B
- 🌐 Simple Flask web interface — no command line needed after setup

## Tech Stack

| Component | Tool |
|---|---|
| Backend | Flask |
| LLM | Groq API (Llama 3.3 70B) |
| Embeddings | HuggingFace (`all-MiniLM-L6-v2`) |
| Vector Store | FAISS |
| PDF Parsing | LangChain (`PyPDFLoader`) |

## How It Works

1. PDF is uploaded and split into chunks (`RecursiveCharacterTextSplitter`)
2. Each chunk is embedded and stored in a FAISS vector index
3. On a user query, the most relevant chunks are retrieved
4. Retrieved context + question is sent to Groq's Llama 3.3 70B for a final answer

## Setup

```bash
# Clone the repo
git clone https://github.com/gmrscw/rag-chatbot.git
cd rag-chatbot

# Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

Create a `.env` file in the project root with your Groq API key:

```
GROQ_API_KEY=your_api_key_here
```

## Run

```bash
python app.py
```

Then open `http://localhost:5000` in your browser, upload a PDF, and start asking questions.

## Project Structure

```
rag-chatbot/
├── app.py              # Main Flask app (loading, chunking, embedding, retrieval, Groq call)
├── templates/
│   └── index.html      # Web interface
├── docs/                # Sample PDFs (notes/PYQs)
├── .env                 # API key (not committed)
└── .gitignore
```

## Future Improvements

- Subject-wise indexing for multiple courses
- Hybrid retrieval (keyword + semantic)
- Chat history / conversation memory
- Deploy to Render.com

## License

This project is for educational purposes.
