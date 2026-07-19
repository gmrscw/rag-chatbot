"""
app.py — Backend for the RAG Chatbot Web Interface

This file connects the existing RAG pipeline (from chunking.py) with a web interface.
Keep this file in the same folder as chunking.py, query.py, rag_answer.py, etc.
so that faiss_index and .env can be used properly.

How to run:
    python app.py
Then open in browser: http://127.0.0.1:5000
"""

from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from groq import Groq
from werkzeug.utils import secure_filename
import os

# Load GROQ_API_KEY from .env
load_dotenv()

app = Flask(__name__)

UPLOAD_FOLDER = "docs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load embeddings model once when server starts
print("Loading embeddings model...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load existing vector store if available
vectorstore = None
current_pdf_name = None

if os.path.exists("faiss_index"):
    print("Loading existing vector store...")
    vectorstore = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )
    current_pdf_name = "OS_notes.pdf"  # default, will update on new upload

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
print("All systems ready. Starting server...")


def build_vectorstore_from_pdf(pdf_path: str):
    """Load PDF, split into chunks, create embeddings and save vector store."""
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(pages)

    new_vectorstore = FAISS.from_documents(chunks, embeddings)
    new_vectorstore.save_local("faiss_index")

    return new_vectorstore, len(pages), len(chunks)


def get_rag_answer(query: str) -> str:
    """Retrieve relevant chunks and get answer from Groq LLM."""
    global vectorstore
    results = vectorstore.similarity_search(query, k=5)
    context = "\n\n".join([r.page_content for r in results])

    # Strong prompt to ensure English responses
    prompt = f"""You are a helpful, accurate AI assistant.
Answer the question based ONLY on the provided context.
Use clear, natural, and professional English.
If the answer is not available in the context, say "This information is not available in the document."

Context:
{context}

Question: {query}

Answer:"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=800
    )
    return response.choices[0].message.content.strip()


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/status")
def status():
    return jsonify({
        "loaded": vectorstore is not None,
        "filename": current_pdf_name
    })


@app.route("/chat", methods=["POST"])
def chat():
    if vectorstore is None:
        return jsonify({"error": "Please upload a PDF first."}), 400

    data = request.get_json()
    query = (data or {}).get("message", "").strip()

    if not query:
        return jsonify({"error": "Question cannot be empty."}), 400

    try:
        answer = get_rag_answer(query)
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"error": "Something went wrong while processing your request."}), 500


@app.route("/upload", methods=["POST"])
def upload():
    global vectorstore, current_pdf_name

    if "pdf" not in request.files:
        return jsonify({"error": "No file was uploaded."}), 400

    file = request.files["pdf"]
    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400
    if not file.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Only PDF files are allowed."}), 400

    filename = secure_filename(file.filename)
    save_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(save_path)

    try:
        new_vs, num_pages, num_chunks = build_vectorstore_from_pdf(save_path)
        vectorstore = new_vs
        current_pdf_name = filename
        
        return jsonify({
            "message": "PDF processed successfully!",
            "filename": filename,
            "pages": num_pages,
            "chunks": num_chunks
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)