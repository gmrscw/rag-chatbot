from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Step 1: PDF load karo (jaisa pehle kiya tha)
pdf_path = "docs/OS_notes.pdf"
loader = PyPDFLoader(pdf_path)
pages = loader.load()
print(f"Total pages loaded: {len(pages)}")

# Step 2: Text splitter banao
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,      # har chunk mein max 1000 characters
    chunk_overlap=100,    # consecutive chunks mein 200 characters overlap
)
chunks = text_splitter.split_documents(pages)
print(f"Total chunks created: {len(chunks)}")

# Step 3: Pehle 2 chunks dikhao (check karne ke liye)
# print("\n--- Chunk 1 ---")
# print(chunks[0].page_content)
# print(f"\nMetadata: {chunks[0].metadata}")

# print("\n--- Chunk 2 ---")
# print(chunks[1].page_content)
# print(f"\nMetadata: {chunks[1].metadata}")

#Step 4: Embeddings + Vector store
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Step 5: Vector store banao aur save karo
vectorstore = FAISS.from_documents(chunks, embeddings)
vectorstore.save_local("faiss_index")
print("Vector store 'faiss_index' folder me save ho gaya!")


