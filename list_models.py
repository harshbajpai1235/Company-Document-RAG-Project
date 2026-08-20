import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

for m in client.models.list():
    name = getattr(m, "name", "")
    # show only Gemini models that can generate content
    if "gemini" in name.lower():
        print(name)