# api.py
from fastapi import FastAPI, UploadFile, File
from core import (
    pdf_to_text,
    pdf_to_images,
    generate_quiz,
    grade_answer,
    digitize_exam
)

app = FastAPI()


# ── DIGITIZE PDF ──
@app.post("/digitize")
async def digitize(file: UploadFile = File(...)):
    pdf_bytes = await file.read()
    return digitize_exam(pdf_bytes)


# ── GENERATE QUIZ ──
@app.post("/generate-quiz")
async def quiz(file: UploadFile = File(...), mode: str = "MCQ", count: int = 5):
    pdf_bytes = await file.read()
    text = pdf_to_text(pdf_bytes)

    result = generate_quiz(text, mode, count)
    return {"quiz": result}


# ── GRADE ANSWER ──
@app.post("/grade")
async def grade(question: str, answer: str):
    return grade_answer(question, answer)
