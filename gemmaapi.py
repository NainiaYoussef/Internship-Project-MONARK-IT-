"""
gemma_api.py — A simple API that exposes the local Gemma model.
The other developer sends a prompt, gets Gemma's response back.
Run with: python -m uvicorn gemma_api:app --reload --port 8000
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from langchain_ollama import ChatOllama

# ── APP ───────────────────────────────────────────────────────
app = FastAPI(
    title="Gemma 3 API",
    description="Send a prompt, get Gemma's response. That's it.",
    version="1.0.0",
)

# ── CORS — allow any frontend to call this ────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── MODEL ─────────────────────────────────────────────────────
llm = ChatOllama(model="gemma3:4b", temperature=0)

# ── REQUEST / RESPONSE MODELS ─────────────────────────────────
class PromptRequest(BaseModel):
    prompt: str
    temperature: Optional[float] = 0.0
    system: Optional[str] = None   # optional system instruction

class PromptResponse(BaseModel):
    response: str
    model: str = "gemma3:4b"

# ── ENDPOINTS ─────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "model": "gemma3:4b",
        "status": "running",
        "usage": "POST /chat with { prompt: '...' }"
    }

@app.get("/health")
def health():
    return {"status": "ok", "model": "gemma3:4b"}


@app.post("/chat", response_model=PromptResponse)
def chat(req: PromptRequest):
    """
    Send any prompt to Gemma and get a response.

    Body:
      - prompt      (required) : the message to send
      - system      (optional) : a system instruction to prepend
      - temperature (optional) : 0.0 = focused, 1.0 = creative. Default 0.0

    Example:
      { "prompt": "What is photosynthesis?", "system": "You are a biology teacher." }
    """
    if not req.prompt.strip():
        raise HTTPException(status_code=400, detail="prompt cannot be empty.")

    try:
        model = ChatOllama(model="gemma3:4b", temperature=req.temperature)

        # Build the full prompt with optional system instruction
        full_prompt = req.prompt
        if req.system:
            full_prompt = f"[SYSTEM]: {req.system}\n\n[USER]: {req.prompt}"

        result = model.invoke(full_prompt)
        return PromptResponse(response=result.content)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model error: {str(e)}")
