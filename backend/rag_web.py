# backend/rag_web.py
import os
from dotenv import load_dotenv
load_dotenv()

from google import genai
from google.genai import types

_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def ask_live(question: str) -> dict:
    """
    Live RAG using Google Search grounding.
    Returns: {"text": str, "citations": [{"title":..., "url":...}, ...]}
    """
    tool = types.Tool(google_search=types.GoogleSearch())
    cfg = types.GenerateContentConfig(tools=[tool])

    resp = _client.models.generate_content(
        model="gemini-2.5-flash",
        contents=question,
        config=cfg,
    )

    text = getattr(resp, "text", None) or "No answer."
    # Try to collect citations (SDK may change; be defensive)
    citations = []
    try:
        cands = getattr(resp, "candidates", []) or []
        gm = getattr(cands[0], "grounding_metadata", None)
        if gm and getattr(gm, "search_entry_point", None):
            # new grounding format (skip)
            pass
        if gm and getattr(gm, "supporting_content", None):
            for sc in gm.supporting_content:
                url = getattr(sc, "uri", None) or getattr(sc, "web", {}).get("uri")
                title = getattr(sc, "title", "") or (url or "")
                if url:
                    citations.append({"title": title, "url": url})
    except Exception:
        pass

    # Fallback: parse markdown links from text if grounding metadata missing
    if not citations:
        import re
        for m in re.finditer(r"\[([^\]]+)\]\((https?://[^\)]+)\)", text):
            citations.append({"title": m.group(1), "url": m.group(2)})

    # De-dupe citations
    seen = set()
    uniq = []
    for c in citations:
        if c["url"] not in seen:
            seen.add(c["url"])
            uniq.append(c)
    return {"text": text, "citations": uniq[:8]}
