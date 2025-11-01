import os, textwrap
from dotenv import load_dotenv
load_dotenv()

from google import genai
from google.genai import types
from backend.rag_local import retrieve

_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

SYS = """You answer strictly using the provided context. 
If the answer is not in the context, say you don't have enough information."""

def answer_local(question: str, k: int = 6) -> dict:
    ctx = retrieve(question, k=k)
    context_text = "\n\n".join([f"[{i+1}] Source: {c['source']}\n{c['text']}" for i, c in enumerate(ctx)])
    prompt = f"{SYS}\n\nContext:\n{context_text}\n\nUser question: {question}\nAnswer with inline [#] citations."

    resp = _client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig()
    )
    text = getattr(resp, "text", "") or "No answer."

    # Prepare simple citations list
    cites = [{"index": i+1, "source": c["source"]} for i, c in enumerate(ctx)]
    return {"text": text, "citations": cites}
