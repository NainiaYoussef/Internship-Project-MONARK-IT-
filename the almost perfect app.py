import streamlit as st
import streamlit.components.v1 as components
import fitz  # PyMuPDF
import re
import base64
import json
from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

# ── SCHEMAS ──────────────────────────────────────────────────
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

# ── CONFIG ────────────────────────────────────────────────────
st.set_page_config(layout="wide", page_title="AI Pedagogical Platform v4")

llm        = ChatOllama(model="gemma3:4b", temperature=0)
quiz_gen   = llm.with_structured_output(Quiz)
grader     = llm.with_structured_output(GradingReport)

for key in ["quiz_data", "digitized_pages", "digitized_questions"]:
    if key not in st.session_state:
        st.session_state[key] = None
if "visual_assets" not in st.session_state:
    st.session_state.visual_assets = []

# ── PDF HELPERS ───────────────────────────────────────────────
def process_pdf(pdf_bytes):
    doc  = fitz.open(stream=pdf_bytes, filetype="pdf")
    text = "".join(p.get_text() for p in doc)
    imgs = []
    for page in doc:
        for img in page.get_images(full=True):
            imgs.append(doc.extract_image(img[0])["image"])
    return text, imgs

def pdf_to_b64_pages(pdf_bytes, dpi=150):
    doc   = fitz.open(stream=pdf_bytes, filetype="pdf")
    mat   = fitz.Matrix(dpi/72, dpi/72)
    pages = []
    for page in doc:
        pix = page.get_pixmap(matrix=mat)
        b64 = base64.b64encode(pix.tobytes("png")).decode()
        pages.append(f"data:image/png;base64,{b64}")
    return pages

# ── GEMMA-POWERED QUESTION EXTRACTOR ─────────────────────────
def extract_questions_with_gemma(full_text: str) -> list:
    """
    Ask Gemma to extract every question from the exam text and return
    a clean JSON list. We chunk the text so small models can handle it.
    """

    chunk_size  = 3000
    overlap     = 300
    chunks      = []
    start       = 0
    while start < len(full_text):
        chunks.append(full_text[start:start + chunk_size])
        start += chunk_size - overlap

    all_questions = []
    seen_stems    = set()

    system_prompt = """You are an expert exam parser.
Given a chunk of exam text, extract every question and return ONLY a JSON array.
No markdown fences, no explanation — raw JSON only.

Each question object must have:
  "number": int (the question number, e.g. 1, 2, 3)
  "stem": str (the full question text exactly as written)
  "type": one of "MCQ" | "Essay" | "Diagram"
    - MCQ: has options A B C D
    - Essay: asks for a written explanation / long answer
    - Diagram: refers to a diagram, image, table, schema, or asks to label/identify something visual
  "options": object like {"A": "...", "B": "...", "C": "...", "D": "..."} — ONLY for MCQ, else null
  "note": str — for Diagram type, describe what kind of visual is referenced (e.g. "cross-section of leaf")

Rules:
- ONLY extract numbered questions (1, 2, 3...). Skip headers, instructions, footers.
- Do NOT invent questions. Extract exactly what is written.
- If a question number appears more than once, keep only the first occurrence.
- Return [] if no questions are found in this chunk.
"""

    for i, chunk in enumerate(chunks):
        prompt = f"{system_prompt}\n\nEXAM CHUNK {i+1}:\n{chunk}\n\nJSON array:"
        try:
            response = llm.invoke(prompt)
            raw = response.content.strip()
            # Strip markdown fences if model added them
            raw = re.sub(r"^```[a-z]*\n?", "", raw)
            raw = re.sub(r"\n?```$", "", raw)
            raw = raw.strip()
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                for q in parsed:
                    num  = q.get("number")
                    stem = q.get("stem", "").strip()
                    if not num or not stem or stem in seen_stems:
                        continue
                    seen_stems.add(stem)
                    all_questions.append({
                        "number":  num,
                        "stem":    stem,
                        "type":    q.get("type", "Essay"),
                        "options": q.get("options"),
                        "note":    q.get("note", ""),
                    })
        except Exception as e:
            # If a chunk fails, skip silently
            continue

    # Deduplicate by number, sort
    seen_nums = set()
    unique    = []
    for q in sorted(all_questions, key=lambda x: x.get("number", 999)):
        if q["number"] not in seen_nums:
            seen_nums.add(q["number"])
            unique.append(q)

    return unique

# ── GENERATE TAB QUIZ RENDERER ────────────────────────────────
def render_quiz(quiz, key_prefix="q"):
    if quiz is None:
        return
    st.header(f"📝 {quiz.title}")
    for i, q in enumerate(quiz.questions):
        with st.container(border=True):
            st.subheader(f"Question {i+1} ({q.type})")
            st.write(q.questionText)
            if q.type == "MultiChoice" and q.options:
                choice = st.radio(
                    "Pick an answer:",
                    [o.text for o in q.options],
                    key=f"{key_prefix}_m{i}",
                    index=None
                )
                if choice:
                    correct = next((o.text for o in q.options if o.isCorrect), None)
                    if choice == correct:
                        st.success("Correct! 🎯")
                    else:
                        st.error(f"Incorrect. The right answer: {correct}")
            else:
                stu_ans = st.text_area("Type your answer here:", key=f"{key_prefix}_a{i}")
                if st.button("Submit & Grade", key=f"{key_prefix}_g{i}"):
                    with st.spinner("Grading..."):
                        report = grader.invoke(
                            f"Reference Answer: {q.sampleAnswer or 'N/A'}\nStudent Answer: {stu_ans}"
                        )
                    c1, c2 = st.columns([1, 4])
                    c1.metric("Score", f"{report.score}/20")
                    c2.success(f"**Feedback:** {report.feedback}")
                    if report.missing_concepts:
                        st.warning(f"**Missing:** {', '.join(report.missing_concepts)}")

# ── DIGITIZER HTML COMPONENT ──────────────────────────────────
def build_digitizer_html(pages_b64: list, questions: list) -> str:
    pages_json     = json.dumps(pages_b64)
    questions_json = json.dumps(questions)

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
html, body {{ height: 100%; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #f0f0ee; overflow: hidden; }}

.shell {{ display: flex; height: 100vh; }}

/* ── LEFT: PDF viewer ── */
.pdf-panel {{
  flex: 1; overflow-y: auto;
  background: #555; padding: 14px;
  display: flex; flex-direction: column; gap: 10px;
}}
.pdf-page-img {{
  width: 100%; display: block;
  border-radius: 3px;
  box-shadow: 0 2px 10px rgba(0,0,0,0.5);
}}

/* ── RIGHT: Answer panel ── */
.answer-panel {{
  width: 340px; flex-shrink: 0;
  display: flex; flex-direction: column;
  background: #fff;
  border-left: 1px solid rgba(0,0,0,0.1);
}}
.panel-header {{
  padding: 12px 14px 10px;
  border-bottom: 1px solid rgba(0,0,0,0.08);
}}
.panel-title {{ font-size: 14px; font-weight: 700; color: #1a1a1a; margin-bottom: 8px; }}
.prog-wrap {{ background: #efefed; border-radius: 4px; height: 4px; overflow: hidden; }}
.prog-fill {{ height: 100%; background: #1a7f5a; border-radius: 4px; transition: width 0.3s; }}
.prog-label {{ font-size: 11px; color: #888; margin-top: 4px; }}

.q-list {{
  flex: 1; overflow-y: auto;
  padding: 8px 10px;
  display: flex; flex-direction: column; gap: 5px;
}}

/* ── Question cards ── */
.q-card {{
  border: 1px solid rgba(0,0,0,0.09);
  border-radius: 10px; padding: 10px 12px;
  background: #fff;
}}
.q-card.answered {{ border-color: #185FA5; background: #f5f9ff; }}
.q-card.submitted-open {{ border-color: #1a7f5a; background: #f0faf5; }}

.q-badge {{
  display: inline-block; font-size: 9px; font-weight: 700;
  padding: 2px 6px; border-radius: 4px; margin-bottom: 5px;
  letter-spacing: 0.4px; text-transform: uppercase;
}}
.badge-mcq    {{ background: #e6f1fb; color: #185FA5; }}
.badge-essay  {{ background: #f0faf5; color: #1a7f5a; }}
.badge-diagram {{ background: #faf0fb; color: #7a3bab; }}

.q-num  {{ font-size: 10px; font-weight: 700; color: #aaa; margin-bottom: 2px; }}
.q-stem {{ font-size: 11.5px; color: #1a1a1a; line-height: 1.5; margin-bottom: 8px; }}
.q-note {{ font-size: 10px; color: #999; font-style: italic; margin-bottom: 6px; }}

/* MCQ options */
.opts {{ display: grid; grid-template-columns: 1fr 1fr; gap: 4px; }}
.opt-btn {{
  display: flex; align-items: flex-start; gap: 5px;
  padding: 5px 7px;
  border: 1px solid rgba(0,0,0,0.1); border-radius: 7px;
  background: #f8f8f6; cursor: pointer;
  font-size: 10.5px; color: #1a1a1a;
  text-align: left; line-height: 1.35;
  transition: background 0.1s, border-color 0.1s;
  font-family: inherit;
}}
.opt-btn:hover:not([disabled]) {{ background: #efefed; border-color: rgba(0,0,0,0.2); }}
.opt-btn[disabled] {{ cursor: default; }}
.opt-btn.selected {{ border-color: #185FA5; background: #e6f1fb; }}
.opt-btn.correct-ans {{ border-color: #1a7f5a !important; background: #e8f7ef !important; }}
.opt-btn.wrong-ans   {{ border-color: #c0392b !important; background: #fcecea !important; }}

.opt-circle {{
  width: 17px; height: 17px; border-radius: 50%; flex-shrink: 0;
  display: flex; align-items: center; justify-content: center;
  font-size: 9px; font-weight: 800;
  background: #fff; border: 1px solid rgba(0,0,0,0.18);
  color: #444; margin-top: 1px;
}}
.opt-btn.selected   .opt-circle {{ background:#185FA5; color:#fff; border-color:#185FA5; }}
.opt-btn.correct-ans .opt-circle {{ background:#1a7f5a; color:#fff; border-color:#1a7f5a; }}
.opt-btn.wrong-ans   .opt-circle {{ background:#c0392b; color:#fff; border-color:#c0392b; }}

/* Essay / open textarea */
.essay-box {{
  width: 100%; font-size: 11px; padding: 7px 9px;
  border: 1px solid rgba(0,0,0,0.12); border-radius: 7px;
  resize: vertical; min-height: 70px;
  background: #f8f8f6; color: #1a1a1a;
  font-family: inherit; line-height: 1.5;
}}
.essay-box:focus {{ outline: none; border-color: #185FA5; }}
.essay-box[disabled] {{ opacity: 0.6; cursor: not-allowed; }}

/* Diagram note box */
.diagram-note {{
  background: #faf0fb; border: 1px dashed #c49ed6;
  border-radius: 7px; padding: 7px 10px;
  font-size: 10.5px; color: #7a3bab; margin-bottom: 6px;
  line-height: 1.4;
}}
.diagram-label {{
  font-size: 10.5px; color: #555; margin-bottom: 4px;
}}

/* Footer */
.panel-footer {{
  padding: 10px 14px;
  border-top: 1px solid rgba(0,0,0,0.08);
  display: flex; flex-direction: column; gap: 7px;
}}
.submit-btn {{
  width: 100%; padding: 10px;
  background: #1a1a1a; color: #fff;
  border: none; border-radius: 8px;
  font-size: 12px; font-weight: 700;
  cursor: pointer; transition: opacity 0.15s;
  font-family: inherit;
}}
.submit-btn:disabled {{ opacity: 0.35; cursor: default; }}
.submit-btn:hover:not(:disabled) {{ opacity: 0.82; }}
.reset-btn {{
  width: 100%; padding: 9px;
  background: transparent; color: #666;
  border: 1px solid rgba(0,0,0,0.15); border-radius: 8px;
  font-size: 12px; cursor: pointer; font-family: inherit;
}}
.reset-btn:hover {{ background: #f5f5f3; }}

.score-card {{
  background: #f5f5f3; border-radius: 8px;
  padding: 10px 14px; text-align: center;
}}
.score-big {{ font-size: 22px; font-weight: 800; color: #1a1a1a; }}
.score-sub {{ font-size: 11px; color: #999; margin-top: 2px; }}
.score-pct {{ font-size: 13px; font-weight: 700; color: #1a7f5a; margin-top: 4px; }}

.remaining-badge {{
  display: inline-flex; align-items: center; justify-content: center;
  width: 16px; height: 16px; border-radius: 50%;
  background: #e8a020; color: #fff;
  font-size: 9px; font-weight: 800; margin-left: 5px;
  vertical-align: middle;
}}
</style>
</head>
<body>
<div class="shell">

  <!-- LEFT: pixel-perfect PDF -->
  <div class="pdf-panel" id="pdfPanel"></div>

  <!-- RIGHT: smart answer panel -->
  <div class="answer-panel">
    <div class="panel-header">
      <div class="panel-title" id="panelTitle">Answer Sheet</div>
      <div class="prog-wrap"><div class="prog-fill" id="progFill" style="width:0%"></div></div>
      <div class="prog-label" id="progLabel">Loading...</div>
    </div>
    <div class="q-list" id="qList"></div>
    <div class="panel-footer" id="footer">
      <button class="submit-btn" id="submitBtn" disabled onclick="handleSubmit()">
        Submit answers
      </button>
    </div>
  </div>
</div>

<script>
const PAGES     = {pages_json};
const QUESTIONS = {questions_json};

let answers   = {{}};   // qNum -> letter (MCQ) or text (essay/diagram)
let submitted = false;

// ── Render PDF pages ─────────────────────────────────────────
const pdfPanel = document.getElementById('pdfPanel');
PAGES.forEach((src, idx) => {{
  const img    = document.createElement('img');
  img.src      = src;
  img.className = 'pdf-page-img';
  img.alt      = 'Page ' + (idx + 1);
  pdfPanel.appendChild(img);
}});

const mcqQs   = QUESTIONS.filter(q => q.type === 'MCQ');
const essayQs = QUESTIONS.filter(q => q.type === 'Essay');
const diagQs  = QUESTIONS.filter(q => q.type === 'Diagram');
const totalAnswerable = QUESTIONS.length;

// ── Progress ─────────────────────────────────────────────────
function updateProgress() {{
  const answered = Object.keys(answers).length;
  const pct      = totalAnswerable > 0 ? Math.round((answered / totalAnswerable) * 100) : 0;
  document.getElementById('progFill').style.width = pct + '%';
  document.getElementById('progLabel').textContent =
    answered + ' of ' + totalAnswerable + ' answered  •  ' +
    mcqQs.length + ' MCQ  ' + essayQs.length + ' Essay  ' + diagQs.length + ' Diagram';
  const remaining = totalAnswerable - answered;
  const title = document.getElementById('panelTitle');
  title.innerHTML = 'Answer Sheet' +
    (remaining > 0 && !submitted
      ? '<span class="remaining-badge">' + remaining + '</span>'
      : '');
  document.getElementById('submitBtn').disabled = answered === 0;
}}

// ── Pick MCQ answer ───────────────────────────────────────────
function pickMCQ(qNum, letter) {{
  if (submitted) return;
  answers[qNum] = letter;
  renderAll();
  updateProgress();
}}

// ── Render all question cards ─────────────────────────────────
function renderAll() {{
  const list = document.getElementById('qList');
  list.innerHTML = '';

  QUESTIONS.forEach(q => {{
    const card = document.createElement('div');
    const sel  = answers[q.number];
    let cls    = 'q-card';
    if (sel) cls += ' answered';
    card.className = cls;

    // type badge
    const badge = document.createElement('span');
    badge.className = 'q-badge badge-' + q.type.toLowerCase();
    badge.textContent = q.type === 'MCQ' ? 'Multiple Choice'
                      : q.type === 'Essay' ? 'Essay / Open Answer'
                      : 'Diagram / Table';
    card.appendChild(badge);

    // question number
    const numDiv = document.createElement('div');
    numDiv.className = 'q-num';
    numDiv.textContent = 'Q' + q.number;
    card.appendChild(numDiv);

    // question stem
    const stemDiv = document.createElement('div');
    stemDiv.className = 'q-stem';
    stemDiv.textContent = q.stem;
    card.appendChild(stemDiv);

    // ── MCQ ──────────────────────────────────────────────────
    if (q.type === 'MCQ' && q.options) {{
      const optsDiv = document.createElement('div');
      optsDiv.className = 'opts';
      Object.entries(q.options).forEach(([letter, text]) => {{
        const btn = document.createElement('button');
        let bcls  = 'opt-btn';
        if (sel === letter) bcls += ' selected';
        btn.className = bcls;
        if (submitted) btn.setAttribute('disabled', true);

        const circle = document.createElement('span');
        circle.className = 'opt-circle';
        circle.textContent = letter;

        const label = document.createElement('span');
        label.textContent = text;

        btn.appendChild(circle);
        btn.appendChild(label);
        btn.onclick = () => pickMCQ(q.number, letter);
        optsDiv.appendChild(btn);
      }});
      card.appendChild(optsDiv);

      if (submitted && sel) {{
        const note = document.createElement('div');
        note.style.cssText = 'margin-top:6px;font-size:10px;color:#888;';
        note.textContent = '✓ Answered: ' + sel + ' — verify with mark scheme';
        card.appendChild(note);
      }}
    }}

    // ── DIAGRAM / TABLE ──────────────────────────────────────
    else if (q.type === 'Diagram') {{
      if (q.note) {{
        const noteBox = document.createElement('div');
        noteBox.className = 'diagram-note';
        noteBox.textContent = '📊 Refers to: ' + q.note;
        card.appendChild(noteBox);
      }}
      const lbl = document.createElement('div');
      lbl.className = 'diagram-label';
      lbl.textContent = 'Your answer (refer to the diagram on the left):';
      card.appendChild(lbl);

      const ta = document.createElement('textarea');
      ta.className = 'essay-box';
      ta.placeholder = 'Write your answer here, referring to the diagram/table/schema shown in the exam...';
      ta.rows = 3;
      if (submitted) ta.setAttribute('disabled', true);
      ta.value = answers[q.number] || '';
      ta.oninput = () => {{
        answers[q.number] = ta.value;
        updateProgress();
      }};
      card.appendChild(ta);
    }}

    // ── ESSAY / OPEN ─────────────────────────────────────────
    else {{
      const ta = document.createElement('textarea');
      ta.className = 'essay-box';
      ta.placeholder = 'Write your answer here...';
      ta.rows = 4;
      if (submitted) ta.setAttribute('disabled', true);
      ta.value = answers[q.number] || '';
      ta.oninput = () => {{
        answers[q.number] = ta.value;
        updateProgress();
      }};
      card.appendChild(ta);
    }}

    list.appendChild(card);
  }});
}}

// ── Submit ────────────────────────────────────────────────────
function handleSubmit() {{
  submitted = true;
  const answered = Object.keys(answers).length;
  const pct = totalAnswerable > 0 ? Math.round((answered / totalAnswerable) * 100) : 0;

  const footer = document.getElementById('footer');
  footer.innerHTML = '';

  const scoreCard = document.createElement('div');
  scoreCard.className = 'score-card';
  scoreCard.innerHTML =
    '<div class="score-big">' + answered + ' / ' + totalAnswerable + '</div>' +
    '<div class="score-sub">questions answered</div>' +
    '<div class="score-pct">' + pct + '% completion rate</div>';
  footer.appendChild(scoreCard);

  const resetBtn = document.createElement('button');
  resetBtn.className = 'reset-btn';
  resetBtn.textContent = '↺  Start over';
  resetBtn.onclick = () => {{
    answers   = {{}};
    submitted = false;
    renderAll();
    updateProgress();
    const f = document.getElementById('footer');
    f.innerHTML = '';
    const sb = document.createElement('button');
    sb.className   = 'submit-btn';
    sb.id          = 'submitBtn';
    sb.disabled    = true;
    sb.textContent = 'Submit answers';
    sb.onclick     = handleSubmit;
    f.appendChild(sb);
    updateProgress();
  }};
  footer.appendChild(resetBtn);

  renderAll();
  updateProgress();
}}

renderAll();
updateProgress();
</script>
</body>
</html>"""

# ── TABS ──────────────────────────────────────────────────────
tab_generate, tab_digitize = st.tabs(["🧠 Generate Quiz", "📄 Digitize Exam PDF"])

# ── TAB 1: GENERATE ───────────────────────────────────────────
with tab_generate:
    with st.sidebar:
        st.title("⚙️ PFE Admin Console")
        pdf  = st.file_uploader("Upload Course PDF", type="pdf", key="gen_pdf")
        mode = st.selectbox("Exam Mode", ["MultiChoice", "Essay", "ShortAnswer", "Mixed Mode"])

        if mode == "Mixed Mode":
            with st.form("mix_form"):
                st.write("### Define Exam Structure")
                c_mcq  = st.number_input("MultiChoice Count", 0, 10, 2)
                c_ess  = st.number_input("Essay Count",       0, 10, 1)
                c_shrt = st.number_input("Short Answer Count",0, 10, 1)
                generate = st.form_submit_button("Generate Custom Mix")
                req_str  = f"EXACTLY {c_mcq} MultiChoice, {c_ess} Essays, and {c_shrt} Short Answers"
        else:
            q_count  = st.slider("Total Questions", 1, 10, 3)
            generate = st.button(f"Generate {mode} Quiz")
            req_str  = f"EXACTLY {q_count} questions of type '{mode}'"

        if pdf and generate:
            with st.spinner("AI is analyzing..."):
                txt, imgs = process_pdf(pdf.read())
                st.session_state.visual_assets = imgs
                prompt = f"""[INST]
SYSTEM: You are a strict academic examiner.
TASK: Generate a quiz based on the text below.
REQUIRED DISTRIBUTION: {req_str}
RULES:
1. For 'MultiChoice', you MUST provide 4 options.
2. For 'Essay', provide a detailed 'sampleAnswer'.
3. For every question, the 'context' field MUST contain a quote from the source.
SOURCE TEXT: {txt[:5000]}
[/INST]"""
                raw_quiz = quiz_gen.invoke(prompt)
                if mode != "Mixed Mode":
                    for q in raw_quiz.questions:
                        q.type = mode
                st.session_state.quiz_data = raw_quiz

    if st.session_state.visual_assets:
        with st.expander("🖼️ Visual Assets"):
            cols = st.columns(3)
            for idx, img in enumerate(st.session_state.visual_assets[:6]):
                cols[idx % 3].image(img, use_container_width=True)

    render_quiz(st.session_state.quiz_data, key_prefix="gen")

    with st.expander("🛠️ Raw AI Output"):
        if st.session_state.quiz_data:
            st.json(st.session_state.quiz_data.dict())

# ── TAB 2: DIGITIZE ───────────────────────────────────────────
with tab_digitize:
    st.header("📄 Digitize an Existing Exam")
    st.write(
        "Upload any exam PDF — pages render **exactly as printed**. "
        "Gemma reads each question and assigns the right answer type: "
        "**MCQ** → click options · **Essay** → text box · **Diagram/Table** → answer referencing the visual."
    )

    dig_pdf = st.file_uploader("Upload Exam PDF", type="pdf", key="dig_pdf")

    if dig_pdf:
        if st.button("🔍 Digitize Exam"):
            pdf_bytes = dig_pdf.read()

            with st.spinner("Step 1/3 — Rendering pages at full quality..."):
                st.session_state.digitized_pages = pdf_to_b64_pages(pdf_bytes, dpi=150)

            with st.spinner("Step 2/3 — Extracting text..."):
                txt, _ = process_pdf(pdf_bytes)

            with st.spinner("Step 3/3 — Gemma is reading and classifying every question..."):
                st.session_state.digitized_questions = extract_questions_with_gemma(txt)

            n      = len(st.session_state.digitized_questions)
            mcq_n  = sum(1 for q in st.session_state.digitized_questions if q['type'] == 'MCQ')
            ess_n  = sum(1 for q in st.session_state.digitized_questions if q['type'] == 'Essay')
            diag_n = sum(1 for q in st.session_state.digitized_questions if q['type'] == 'Diagram')
            st.success(
                f"✅ {n} questions found — "
                f"{mcq_n} MCQ · {ess_n} Essay · {diag_n} Diagram/Table"
            )

    if st.session_state.digitized_pages and st.session_state.digitized_questions:
        html = build_digitizer_html(
            st.session_state.digitized_pages,
            st.session_state.digitized_questions
        )
        components.html(html, height=900, scrolling=False)

    with st.expander("🛠️ Raw Extracted Questions (debug)"):
        if st.session_state.digitized_questions:
            st.json(st.session_state.digitized_questions)
