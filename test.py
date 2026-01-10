# check_models.py
import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GENAI_API_KEY")

if not api_key:
    print("❌ Error: GENAI_API_KEY not found in .env")
else:
    genai.configure(api_key=api_key)
    print("🔍 Checking available models for your API key...")
    try:
        found_any = False
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                print(f" - {m.name}")
                found_any = True
        if not found_any:
            print("⚠️ No models found. Your API Key might be restricted or invalid.")
    except Exception as e:
        print(f"❌ Connection failed: {e}")