from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq

# .env file se API key load karna
load_dotenv()

# Check key mili ya nahi
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    print("ERROR: GROQ_API_KEY nahi mili! .env file check karo.")
else:
    print("API key mil gayi, ab Groq ko test kar rahe hain...")

    # Groq model banao
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        groq_api_key=api_key
    )

    # Simple test question
    response = llm.invoke("Namaste! Ek line mein bata, tum kaun ho?")
    print("\nGroq ka jawab:")
    print(response.content)
