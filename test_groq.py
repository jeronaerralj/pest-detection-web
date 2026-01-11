# check_groq_models.py
import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    print("❌ Error: GROQ_API_KEY not found in .env")
else:
    print("🔍 Checking available models for your Groq API key...")
    try:
        # Initialize the Groq client
        client = Groq(api_key=api_key)
        
        # Fetch list of models
        models = client.models.list()
        
        found_any = False
        # Iterate through the data list provided by Groq
        for m in models.data:
            print(f" - {m.id}")
            found_any = True
            
        if not found_any:
            print("⚠️ No models found. Your API Key might be restricted or invalid.")
            
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        if "401" in str(e):
            print("💡 Tip: Check if your API Key is correct in the .env file.")