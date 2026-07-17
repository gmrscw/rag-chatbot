from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from groq import Groq
import os

#Step 1: .env se API key load karo
load_dotenv()

#Step 2: Same embedding model load karo (jo chunking.py me use kiya tha)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

#Step 3: saved vector store load karo
vectorstore = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

#Step 4: Groq client banao
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

#Step 5: Query lo aur relevant chunks retrieve karo
query = "Deadlock kya hota hai?"
results = vectorstore.similarity_search(query, k=3)

#Step 6: Retrieved chunks ko ek context string me jodo
context = "\n\n".join([r.page_content for r in results])

#Step 7: Prompt banao(context + query LLM ko bhejne ke liye)
prompt = f"""Neeche diye gaye context ke basis par question ka answer do.
Agar context me answer nahi hai to bol do "Is context me ye information nahi hai."

Context:
{context}

Question: {query}
Answer: """

#Step 8: Groq LLM ko call karo
response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {"role": "user", "content": prompt}
    ]
)

#Step 9: Answer print karo
print("Answer:\n")
print(response.choices[0].message.content)