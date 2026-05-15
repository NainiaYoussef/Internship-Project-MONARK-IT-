# core.py
import fitz
import base64
import re
from typing import List, Dict
from langchain_ollama import ChatOllama

# ── LOAD MODEL ──
llm = ChatOllama(model="gemma3:4b", temperature=0)

# ── DIGITIZE PDF ──
def pdf_to_text(pdf_bytes: bytes) -> str:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    return "\n".join(page.get_text() for page in doc)

def pdf_to_images(pdf_bytes: bytes):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    images = []
    for page in doc:
        pix = page.get_pixmap()
        images.append(base64.b64encode(pix.tobytes()).decode())
    return images


# ── QUIZ GENERATION ──
def generate_quiz(text: str, mode: str, count: int):
    prompt = f"""
You are an educational AI.

Generate {count} questions.
Type: {mode}

Text:
{text[:5000]}

Return JSON only:
"""
    return llm.invoke(prompt)


# ── GRADING ──
def grade_answer(question: str, answer: str):
    prompt = f"""
Grade this answer.

Question: {question}
Answer: {answer}

Return score + feedback.
"""
    return llm.invoke(prompt)


# ── DIGITIZE EXAM (YOUR PARSER WRAPPED) ──
def digitize_exam(pdf_bytes: bytes):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    text = []
    for page in doc:
        text.append(page.get_text())

    return {
        "raw_text": "\n".join(text),
        "pages": len(doc)
    }
