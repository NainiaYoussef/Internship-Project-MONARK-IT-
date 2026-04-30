import streamlit as st
import json
import fitz  # PyMuPDF
from io import BytesIO
from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama

# --- 1. SCHEMAS (Data Structure) ---
class Option(BaseModel):
    text: str
    isCorrect: bool

class Question(BaseModel):
    questionText: str
    type: str = Field(description="Must be 'MultiChoice', 'ShortAnswer', or 'Essay'")
    options: Optional[List[Option]] = None
    sampleAnswer: Optional[str] = None
    context: str = Field(description="The exact quote from the PDF")

class Quiz(BaseModel):
    title: str
    questions: List[Question]

class GradingReport(BaseModel):
    score: int = Field(description="Score out of 20")
    feedback: str = Field(description="Pedagogical feedback")
    missing_concepts: List[str] = Field(description="Points missed")

# --- 2. CONFIG & LLM ---
st.set_page_config(layout="wide", page_title="AI Pedagogical Platform v3")

# Using ChatOllama for native Structured Output
llm = ChatOllama(model="gemma3:4b", temperature=0)
quiz_gen = llm.with_structured_output(Quiz)
grader = llm.with_structured_output(GradingReport)

if "quiz_data" not in st.session_state: st.session_state.quiz_data = None
if "visual_assets" not in st.session_state: st.session_state.visual_assets = []

# --- 3. EXTRACTION ENGINE ---
def process_pdf(pdf_bytes):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    text = "".join([page.get_text() for page in doc])
    images = []
    for page in doc:
        for img in page.get_images(full=True):
            images.append(doc.extract_image(img[0])["image"])
    return text, images

# --- 4. SIDEBAR (The Formulaire) ---
with st.sidebar:
    st.title("⚙️ PFE Admin Console")
    pdf = st.file_uploader("Upload Course PDF", type="pdf")
    mode = st.selectbox("Exam Mode", ["MultiChoice", "Essay", "ShortAnswer", "Mixed Mode"])
    
    # Logic for Form vs Slider
    if mode == "Mixed Mode":
        with st.form("mix_form"):
            st.write("### Define Exam Structure")
            c_mcq = st.number_input("MultiChoice Count", 0, 10, 2)
            c_ess = st.number_input("Essay Count", 0, 10, 1)
            c_shrt = st.number_input("Short Answer Count", 0, 10, 1)
            generate = st.form_submit_button("Generate Custom Mix")
            # Strict requirement string for the prompt
            req_str = f"EXACTLY {c_mcq} MultiChoice, {c_ess} Essays, and {c_shrt} Short Answers"
    else:
        q_count = st.slider("Total Questions", 1, 10, 3)
        generate = st.button(f"Generate {mode} Quiz")
        req_str = f"EXACTLY {q_count} questions of type '{mode}'"

    # TRIGGER GENERATION
    if pdf and generate:
        with st.spinner("AI is analyzing and enforcing strict formatting..."):
            txt, imgs = process_pdf(pdf.read())
            st.session_state.visual_assets = imgs
            
            # STERN PROMPT FOR GEMMA 3
            prompt = f"""
            [INST]
            SYSTEM: You are a strict academic examiner.
            TASK: Generate a quiz based on the text below.
            REQUIRED DISTRIBUTION: {req_str}
            RULES:
            1. For 'MultiChoice', you MUST provide 4 options.
            2. For 'Essay', provide a detailed 'sampleAnswer'.
            3. For every question, the 'context' field MUST contain the quote from the source.
            
            SOURCE TEXT: {txt[:5000]}
            [/INST]
            """
            
            raw_quiz = quiz_gen.invoke(prompt)
            
            # TYPE GUARD: We manually force the AI to respect our choice
            if mode != "Mixed Mode":
                for q in raw_quiz.questions:
                    q.type = mode
            
            st.session_state.quiz_data = raw_quiz

# --- 5. MAIN UI ---
if st.session_state.quiz_data:
    quiz = st.session_state.quiz_data
    st.header(f"📝 {quiz.title}")
    
    # Visual Assets Expander
    if st.session_state.visual_assets:
        with st.expander("🖼️ Visual Assets (Extracted Schemas/Photos)"):
            cols = st.columns(3)
            for idx, img in enumerate(st.session_state.visual_assets[:6]):
                cols[idx%3].image(img, use_container_width=True)

    # Rendering the Quiz
    for i, q in enumerate(quiz.questions):
        with st.container(border=True):
            st.subheader(f"Question {i+1} ({q.type})")
            st.write(q.questionText)
            
            # MCQ RENDERER
            if q.type == "MultiChoice" and q.options:
                choice = st.radio("Pick an answer:", [o.text for o in q.options], key=f"m{i}", index=None)
                if choice:
                    correct = next((o.text for o in q.options if o.isCorrect), "Not Found")
                    if choice == correct: st.success("Correct! 🎯")
                    else: st.error(f"Incorrect. The right answer: {correct}")
                    st.info(f"**Evidence:** {q.context}")

            # ESSAY / SHORT ANSWER RENDERER
            else:
                stu_ans = st.text_area("Type your answer here:", key=f"a{i}")
                if st.button("Submit & Grade", key=f"g{i}"):
                    # Semantic Grading Call
                    report = grader.invoke(f"Ref: {q.sampleAnswer}\nStudent Answer: {stu_ans}")
                    c1, c2 = st.columns([1, 4])
                    c1.metric("Score", f"{report.score}/20")
                    c2.success(f"**Feedback:** {report.feedback}")
                    if report.missing_concepts:
                        st.warning(f"**Missing:** {', '.join(report.missing_concepts)}")
                    st.info(f"**Source Quote:** {q.context}")

# Footer debug
with st.expander("🛠️ Raw AI Output"):
    if st.session_state.quiz_data:
        st.json(st.session_state.quiz_data.dict())
