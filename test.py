from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI

# .env file se API key load karo
load_dotenv()

# Check karo key mili ya nahi
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    print("ERROR: API key nahi mili! .env file check karo.")
else:
    print("API key mil gayi, ab Gemini ko test kar rahe hain...")

    # Gemini model banao
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=api_key
    )

    # Simple test question
    response = llm.invoke("Namaste! Ek line mein bata, tum kaun ho?")
    print("\nGemini ka jawab:")
    print(response.content)