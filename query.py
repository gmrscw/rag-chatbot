from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

#Step 1: Same Embedding model load karo (jo chunking.py me use kiya tha)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

#Step 2: Saved vector store load karo
vectorstore = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True    # local file trust karne ke liye zaroori hai
)

#Step 3: Query chalao
query = "process kya hota hai?"
results = vectorstore.similarity_search(query, k=3)

print(f"\nQuery: {query}\n")
for i,r in enumerate(results):
    print(f"--- Result {i+1} ---")
    print(r.page_content)
    print(f"Metadata: {r.metadata}")
    print()