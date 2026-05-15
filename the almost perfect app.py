import streamlit as st
import streamlit.components.v1 as components
import fitz  # PyMuPDF
import re, base64, json
from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama

# ── SCHEMAS ───────────────────────────────────────────────────
class Option(BaseModel):
    text: str
    isCorrect: bool

class Question(BaseModel):
    questionText: str
    type: str = Field(description="'MultiChoice', 'ShortAnswer', or 'Essay'")
    options: Optional[List[Option]] = None
    sampleAnswer: Optional[str] = None
    context: str

class Quiz(BaseModel):
    title: str
    questions: List[Question]

class GradingReport(BaseModel):
    score: int
    feedback: str
    missing_concepts: List[str]

# ── CONFIG ────────────────────────────────────────────────────
st.set_page_config(layout="wide", page_title="AI Pedagogical Platform")
llm      = ChatOllama(model="gemma3:4b", temperature=0)
quiz_gen = llm.with_structured_output(Quiz)
grader   = llm.with_structured_output(GradingReport)

for k in ["quiz_data","digitized_pages","digitized_questions","visual_assets"]:
    if k not in st.session_state:
        st.session_state[k] = None if k != "visual_assets" else []

# ── PDF HELPERS ───────────────────────────────────────────────
def process_pdf(pdf_bytes):
    doc  = fitz.open(stream=pdf_bytes, filetype="pdf")
    text = "".join(p.get_text() for p in doc)
    imgs = [doc.extract_image(i[0])["image"]
            for p in doc for i in p.get_images(full=True)]
    return text, imgs

def pdf_to_b64_pages(pdf_bytes, dpi=150):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    mat = fitz.Matrix(dpi/72, dpi/72)
    out = []
    for page in doc:
        pix = page.get_pixmap(matrix=mat)
        out.append("data:image/png;base64," +
                   base64.b64encode(pix.tobytes("png")).decode())
    return out

def extract_figure_coords(pdf_bytes, dpi=150):
    doc    = fitz.open(stream=pdf_bytes, filetype="pdf")
    scale  = dpi / 72
    result = {}
    for i, page in enumerate(doc):
        rects = []
        for img_info in page.get_images(full=True):
            for r in page.get_image_rects(img_info[0]):
                rects.append({
                    "x": round(r.x0 * scale), "y": round(r.y0 * scale),
                    "w": round((r.x1-r.x0)*scale), "h": round((r.y1-r.y0)*scale),
                })
        if rects:
            result[i] = rects
    return result

# ── BULLETPROOF CAMBRIDGE-FORMAT PARSER ───────────────────────
# Handles two question formats found in this PDF:
#   Format A: '1\t'  alone on a line, then stem on next lines
#   Format B: '10 \t The diagrams show...'  number+tab+stem on same line

JUNK_RE = re.compile(
    r'^\s*$|^©|^\[Turn over|^0970|^This document|^Permission|^Cambridge Assessment'
    r'|^UCLES|^You must|^You will|^You may|^INSTRUCTIONS|^INFORMATION'
    r'|^There are forty|^Answer all|^Write your|^Write in soft|^Soft clean|^Soft pencil'
    r'|^Follow the|^Do not|^For each question|^The total mark|^Each correct'
    r'|^Any rough|\*0123456789\*|^BIOLOGY|^Paper 1|^SPECIMEN|^key:|^= yes|^= no'
    r'|^\uf0fc|^\uf0fb|^●',
    re.IGNORECASE
)
# Q_ALONE: '1\t' or '23\t' — just number+tab, nothing else
Q_ALONE  = re.compile(r'^(\d{1,2})\t\s*$')
# Q_INLINE: '10 \t Some stem...' — number+optional-space+tab+stem text
Q_INLINE = re.compile(r'^(\d{1,2})\s*\t\s*(.+)$')
# OPT_ALONE: 'A\t' or 'A ' alone
OPT_ALONE   = re.compile(r'^([ABCD])\t?\s*$')
# OPT_INLINE: 'A  some text' (2+ spaces after letter)
OPT_INLINE  = re.compile(r'^([ABCD])\s{2,}(.+)$')

DIAG_KW = re.compile(
    r'diagram|table|graph|photograph|photomicrograph|figure|apparatus|cross.section'
    r'|shows a|shown|data shows|information shows|timeline|food web',
    re.IGNORECASE
)

def _is_junk(s):
    s = s.strip()
    return not s or bool(JUNK_RE.match(s)) or s in ('\t','●','\uf0fc','\uf0fb')

def parse_cambridge_pdf(pdf_bytes: bytes) -> list:
    doc        = fitz.open(stream=pdf_bytes, filetype="pdf")
    page_lines = [(i, page.get_text().splitlines()) for i, page in enumerate(doc)]

    questions = []
    seen      = set()

    for page_idx, raw in page_lines:
        lines = raw
        i     = 0
        while i < len(lines):
            line = lines[i]

            m_inline = Q_INLINE.match(line)
            m_alone  = Q_ALONE.match(line)

            # Format B: number+tab+stem on same line
            if m_inline and not m_alone:
                q_num = int(m_inline.group(1))
                if q_num in seen or not (1 <= q_num <= 60):
                    i += 1; continue
                stem_parts = [m_inline.group(2).strip()]
                i += 1

            # Format A: number+tab alone
            elif m_alone:
                q_num = int(m_alone.group(1))
                if q_num in seen or not (1 <= q_num <= 60):
                    i += 1; continue
                stem_parts = []
                i += 1

            else:
                i += 1; continue

            # Collect stem lines until option A or next question
            while i < len(lines):
                nl  = lines[i]
                ns  = nl.strip()
                if OPT_ALONE.match(nl) or OPT_INLINE.match(nl):
                    break
                m_next = Q_ALONE.match(nl) or (Q_INLINE.match(nl) and not Q_ALONE.match(nl))
                if m_next:
                    cand = int(m_next.group(1))
                    if cand > q_num and cand not in seen:
                        break
                if not _is_junk(ns):
                    stem_parts.append(ns)
                i += 1

            stem = ' '.join(p for p in stem_parts if p).strip()
            if not stem or len(stem) < 4:
                continue

            # Collect A B C D options
            options = {}
            while i < len(lines) and len(options) < 4:
                nl = lines[i]
                m_oa = OPT_ALONE.match(nl)
                m_oi = OPT_INLINE.match(nl)
                if m_oa:
                    letter = m_oa.group(1)
                    i += 1
                    # grab the very next non-junk line as option text
                    while i < len(lines) and _is_junk(lines[i].strip()):
                        i += 1
                    if i < len(lines):
                        t = lines[i].strip()
                        # make sure it's not another option or question
                        if t and not OPT_ALONE.match(lines[i]) and \
                           not Q_ALONE.match(lines[i]) and not Q_INLINE.match(lines[i]):
                            options[letter] = t
                            i += 1
                elif m_oi:
                    options[m_oi.group(1)] = m_oi.group(2).strip()
                    i += 1
                elif Q_ALONE.match(nl) or Q_INLINE.match(nl):
                    break
                elif _is_junk(nl.strip()):
                    i += 1
                else:
                    break  # non-option non-junk line after stem = stop

            seen.add(q_num)

            has_opts = len(options) == 4
            is_diag  = bool(DIAG_KW.search(stem))
            qtype    = 'MCQ' if has_opts else ('Diagram' if is_diag else 'Essay')

            questions.append({
                'number':  q_num,
                'stem':    stem,
                'type':    qtype,
                'options': options if has_opts else None,
                'page':    page_idx,
            })

    return sorted(questions, key=lambda x: x['number'])

# ── GENERATE TAB RENDERER ─────────────────────────────────────
def render_quiz(quiz, key_prefix="q"):
    if not quiz: return
    st.header(f"📝 {quiz.title}")
    for i, q in enumerate(quiz.questions):
        with st.container(border=True):
            st.subheader(f"Q{i+1} ({q.type})")
            st.write(q.questionText)
            if q.type == "MultiChoice" and q.options:
                choice = st.radio("Answer:", [o.text for o in q.options],
                                  key=f"{key_prefix}_m{i}", index=None)
                if choice:
                    correct = next((o.text for o in q.options if o.isCorrect), None)
                    if choice == correct: st.success("Correct! 🎯")
                    else: st.error(f"Wrong. Correct: {correct}")
            else:
                ans = st.text_area("Your answer:", key=f"{key_prefix}_a{i}")
                if st.button("Grade", key=f"{key_prefix}_g{i}"):
                    with st.spinner("Grading..."):
                        r = grader.invoke(f"Ref: {q.sampleAnswer or 'N/A'}\nAnswer: {ans}")
                    c1,c2 = st.columns([1,4])
                    c1.metric("Score", f"{r.score}/20")
                    c2.success(f"**Feedback:** {r.feedback}")
                    if r.missing_concepts:
                        st.warning(f"**Missing:** {', '.join(r.missing_concepts)}")

# ── HTML DIGITIZER COMPONENT ──────────────────────────────────
def build_digitizer_html(pages_b64, questions, figure_coords):
    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
html,body{{height:100%;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;overflow:hidden}}
.shell{{display:flex;height:100vh}}

.pdf-panel{{flex:1;overflow-y:auto;background:#4a4a4a;padding:12px;display:flex;flex-direction:column;gap:10px}}
.page-wrap{{position:relative;width:100%}}
.pdf-page-img{{width:100%;display:block;border-radius:3px;box-shadow:0 2px 12px rgba(0,0,0,.55)}}
.page-lbl{{position:absolute;top:6px;left:6px;background:rgba(0,0,0,.5);color:#fff;font-size:10px;padding:2px 8px;border-radius:4px;pointer-events:none}}
.fig-hl{{position:absolute;pointer-events:none;border:3px solid #f5c542;border-radius:4px;background:rgba(245,197,66,.15);display:none}}
@keyframes flashBorder{{0%,100%{{border-color:#f5c542;box-shadow:0 0 8px #f5c542}}50%{{border-color:#ff8c00;box-shadow:0 0 16px #ff8c00}}}}
.fig-hl.active{{display:block;animation:flashBorder 0.7s ease 4}}

.answer-panel{{width:348px;flex-shrink:0;display:flex;flex-direction:column;background:#fff;border-left:1px solid rgba(0,0,0,.1)}}
.panel-header{{padding:12px 14px 10px;border-bottom:1px solid rgba(0,0,0,.08)}}
.panel-title{{font-size:14px;font-weight:700;color:#111;margin-bottom:8px}}
.prog-wrap{{background:#ebebeb;border-radius:4px;height:4px;overflow:hidden}}
.prog-fill{{height:100%;background:#1a7f5a;border-radius:4px;transition:width .3s}}
.prog-lbl{{font-size:11px;color:#999;margin-top:4px}}
.q-list{{flex:1;overflow-y:auto;padding:8px 10px;display:flex;flex-direction:column;gap:5px}}

.q-card{{border:1px solid rgba(0,0,0,.09);border-radius:10px;padding:10px 12px;background:#fff}}
.q-card.answered{{border-color:#185FA5;background:#f5f9ff}}
.q-badge{{display:inline-block;font-size:9px;font-weight:700;padding:2px 6px;border-radius:4px;margin-bottom:5px;letter-spacing:.5px;text-transform:uppercase}}
.b-mcq{{background:#e6f1fb;color:#185FA5}}
.b-essay{{background:#f0faf5;color:#1a7f5a}}
.b-diagram{{background:#faf0fb;color:#7a3bab}}
.q-num{{font-size:10px;font-weight:700;color:#bbb;margin-bottom:2px}}
.q-stem{{font-size:11.5px;color:#111;line-height:1.55;margin-bottom:8px}}

.opts{{display:grid;grid-template-columns:1fr 1fr;gap:4px}}
.opt-btn{{display:flex;align-items:flex-start;gap:5px;padding:5px 7px;border:1px solid rgba(0,0,0,.1);border-radius:7px;background:#f8f8f6;cursor:pointer;font-size:10.5px;color:#111;text-align:left;line-height:1.35;transition:background .1s,border-color .1s;font-family:inherit;width:100%}}
.opt-btn:hover:not([disabled]){{background:#ebebeb;border-color:rgba(0,0,0,.22)}}
.opt-btn[disabled]{{cursor:default}}
.opt-btn.sel{{border-color:#185FA5;background:#deeeff}}
.opt-circle{{width:17px;height:17px;border-radius:50%;flex-shrink:0;display:flex;align-items:center;justify-content:center;font-size:9px;font-weight:800;background:#fff;border:1px solid rgba(0,0,0,.18);color:#444;margin-top:1px}}
.opt-btn.sel .opt-circle{{background:#185FA5;color:#fff;border-color:#185FA5}}

.ans-box{{width:100%;font-size:11px;padding:7px 9px;border:1px solid rgba(0,0,0,.12);border-radius:7px;resize:vertical;min-height:68px;background:#f8f8f6;color:#111;font-family:inherit;line-height:1.5}}
.ans-box:focus{{outline:none;border-color:#185FA5}}
.ans-box[disabled]{{opacity:.6;cursor:not-allowed}}

.diag-ref{{background:#faf0fb;border:1px dashed #c49ed6;border-radius:8px;padding:8px 10px;margin-bottom:7px;display:flex;align-items:center;justify-content:space-between;gap:8px}}
.diag-ref-title{{font-size:10px;font-weight:700;color:#7a3bab;text-transform:uppercase;letter-spacing:.5px}}
.view-btn{{display:inline-flex;align-items:center;gap:5px;padding:5px 11px;background:#7a3bab;color:#fff;border:none;border-radius:6px;font-size:10.5px;font-weight:700;cursor:pointer;font-family:inherit;transition:opacity .15s;white-space:nowrap;flex-shrink:0}}
.view-btn:hover{{opacity:.82}}
.diag-lbl{{font-size:10.5px;color:#555;margin-bottom:4px;margin-top:5px}}

.panel-footer{{padding:10px 14px;border-top:1px solid rgba(0,0,0,.08);display:flex;flex-direction:column;gap:7px}}
.submit-btn{{width:100%;padding:10px;background:#111;color:#fff;border:none;border-radius:8px;font-size:12px;font-weight:700;cursor:pointer;transition:opacity .15s;font-family:inherit}}
.submit-btn:disabled{{opacity:.32;cursor:default}}
.submit-btn:hover:not(:disabled){{opacity:.8}}
.reset-btn{{width:100%;padding:9px;background:transparent;color:#555;border:1px solid rgba(0,0,0,.15);border-radius:8px;font-size:12px;cursor:pointer;font-family:inherit}}
.reset-btn:hover{{background:#f5f5f3}}
.score-card{{background:#f5f5f3;border-radius:8px;padding:10px 14px;text-align:center}}
.score-big{{font-size:22px;font-weight:800;color:#111}}
.score-sub{{font-size:11px;color:#999;margin-top:2px}}
.score-pct{{font-size:13px;font-weight:700;color:#1a7f5a;margin-top:4px}}
.rem-badge{{display:inline-flex;align-items:center;justify-content:center;width:16px;height:16px;border-radius:50%;background:#e8a020;color:#fff;font-size:9px;font-weight:800;margin-left:5px;vertical-align:middle}}
</style>
</head>
<body>
<div class="shell">
  <div class="pdf-panel" id="pdfPanel"></div>
  <div class="answer-panel">
    <div class="panel-header">
      <div class="panel-title" id="panelTitle">Answer Sheet</div>
      <div class="prog-wrap"><div class="prog-fill" id="progFill" style="width:0%"></div></div>
      <div class="prog-lbl" id="progLbl">—</div>
    </div>
    <div class="q-list" id="qList"></div>
    <div class="panel-footer" id="footer">
      <button class="submit-btn" id="submitBtn" disabled onclick="handleSubmit()">Submit answers</button>
    </div>
  </div>
</div>

<script>
const PAGES   = {json.dumps(pages_b64)};
const QS      = {json.dumps(questions)};
const FIGURES = {json.dumps({str(k): v for k,v in figure_coords.items()})};

let answers = {{}};
let submitted = false;
const pageImgs = [];
const pageHls  = [];

/* ── Build PDF pages ── */
const pdfPanel = document.getElementById('pdfPanel');
PAGES.forEach((src, idx) => {{
  const wrap = document.createElement('div');
  wrap.className = 'page-wrap';

  const img = document.createElement('img');
  img.src = src; img.className = 'pdf-page-img';
  img.id = 'pdfPage' + idx; img.alt = 'Page '+(idx+1);

  const lbl = document.createElement('div');
  lbl.className = 'page-lbl';
  lbl.textContent = 'p.'+(idx+1);

  const hl = document.createElement('div');
  hl.className = 'fig-hl'; hl.id = 'hl'+idx;

  wrap.appendChild(img); wrap.appendChild(lbl); wrap.appendChild(hl);
  pdfPanel.appendChild(wrap);
  pageImgs.push(img); pageHls.push(hl);
}});

/* ── Scroll + highlight ── */
function scrollAndHighlight(pageIdx) {{
  const img = document.getElementById('pdfPage'+pageIdx);
  if (!img) return;
  img.scrollIntoView({{behavior:'smooth', block:'start'}});

  pageHls.forEach(h => {{ h.classList.remove('active'); h.style.display='none'; }});

  const figs = FIGURES[String(pageIdx)];
  if (figs && figs.length > 0) {{
    // pick the largest figure (by area)
    const fig = figs.reduce((a,b) => (b.w*b.h > a.w*a.h ? b : a));
    const imgEl = pageImgs[pageIdx];
    const sx = imgEl.clientWidth  / imgEl.naturalWidth;
    const sy = imgEl.clientHeight / imgEl.naturalHeight;
    const hl = pageHls[pageIdx];
    hl.style.left   = (fig.x*sx)+'px';
    hl.style.top    = (fig.y*sy)+'px';
    hl.style.width  = (fig.w*sx)+'px';
    hl.style.height = (fig.h*sy)+'px';
    hl.style.display = 'block';
    hl.classList.add('active');
    setTimeout(()=>{{ hl.classList.remove('active'); hl.style.display='none'; }}, 3500);
  }}
}}

const mcqQs  = QS.filter(q=>q.type==='MCQ');
const diagQs = QS.filter(q=>q.type==='Diagram');
const essQs  = QS.filter(q=>q.type==='Essay');
const total  = QS.length;

function updateProgress() {{
  const ans = Object.keys(answers).length;
  const pct = total>0 ? Math.round(ans/total*100) : 0;
  document.getElementById('progFill').style.width = pct+'%';
  document.getElementById('progLbl').textContent =
    ans+' of '+total+' answered  ·  '+
    mcqQs.length+' MCQ  '+essQs.length+' Essay  '+diagQs.length+' Diagram';
  const rem = total-ans;
  document.getElementById('panelTitle').innerHTML = 'Answer Sheet'+
    (rem>0&&!submitted?'<span class="rem-badge">'+rem+'</span>':'');
  document.getElementById('submitBtn').disabled = ans===0;
}}

function pickMCQ(num, letter) {{
  if (submitted) return;
  answers[num] = letter;
  renderAll(); updateProgress();
}}

function renderAll() {{
  const list = document.getElementById('qList');
  list.innerHTML = '';
  QS.forEach(q => {{
    const sel  = answers[q.number];
    const card = document.createElement('div');
    card.className = 'q-card'+(sel?' answered':'');

    const badge = document.createElement('span');
    badge.className = 'q-badge '+(q.type==='MCQ'?'b-mcq':q.type==='Essay'?'b-essay':'b-diagram');
    badge.textContent = q.type==='MCQ'?'Multiple Choice':q.type==='Essay'?'Essay / Open':'Diagram / Table / Schema';
    card.appendChild(badge);

    const numDiv = document.createElement('div');
    numDiv.className = 'q-num';
    numDiv.textContent = 'Q'+q.number+'  ·  page '+(q.page+1);
    card.appendChild(numDiv);

    const stemDiv = document.createElement('div');
    stemDiv.className = 'q-stem';
    stemDiv.textContent = q.stem;
    card.appendChild(stemDiv);

    if (q.type==='MCQ' && q.options) {{
      const optsDiv = document.createElement('div');
      optsDiv.className = 'opts';
      Object.entries(q.options).forEach(([letter,text]) => {{
        const btn = document.createElement('button');
        btn.className = 'opt-btn'+(sel===letter?' sel':'');
        if (submitted) btn.disabled = true;
        const circle = document.createElement('span');
        circle.className = 'opt-circle'; circle.textContent = letter;
        const lbl = document.createElement('span'); lbl.textContent = text;
        btn.appendChild(circle); btn.appendChild(lbl);
        btn.onclick = ()=>pickMCQ(q.number, letter);
        optsDiv.appendChild(btn);
      }});
      card.appendChild(optsDiv);
      if (submitted&&sel) {{
        const n=document.createElement('div');
        n.style.cssText='margin-top:6px;font-size:10px;color:#888';
        n.textContent='✓ Your answer: '+sel+' — verify with mark scheme';
        card.appendChild(n);
      }}
    }}

    else if (q.type==='Diagram') {{
      const refBox = document.createElement('div');
      refBox.className = 'diag-ref';
      const refTitle = document.createElement('div');
      refTitle.className='diag-ref-title';
      refTitle.textContent='📌 Refers to visual on page '+(q.page+1);
      refBox.appendChild(refTitle);
      const vBtn = document.createElement('button');
      vBtn.className='view-btn';
      vBtn.innerHTML='<svg width="12" height="12" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="2"><path d="M1 8s3-5 7-5 7 5 7 5-3 5-7 5-7-5-7-5z"/><circle cx="8" cy="8" r="2"/></svg> View in PDF';
      vBtn.onclick=()=>scrollAndHighlight(q.page);
      refBox.appendChild(vBtn);
      card.appendChild(refBox);
      const lbl=document.createElement('div');
      lbl.className='diag-lbl';
      lbl.textContent='Your answer (see diagram highlighted on the left):';
      card.appendChild(lbl);
      const ta=document.createElement('textarea');
      ta.className='ans-box';
      ta.placeholder='Write your answer referring to the diagram on page '+(q.page+1)+'…';
      ta.rows=3;
      if(submitted) ta.disabled=true;
      ta.value=answers[q.number]||'';
      ta.oninput=()=>{{answers[q.number]=ta.value;updateProgress();}};
      card.appendChild(ta);
    }}

    else {{
      const ta=document.createElement('textarea');
      ta.className='ans-box'; ta.placeholder='Write your answer here…'; ta.rows=4;
      if(submitted) ta.disabled=true;
      ta.value=answers[q.number]||'';
      ta.oninput=()=>{{answers[q.number]=ta.value;updateProgress();}};
      card.appendChild(ta);
    }}

    list.appendChild(card);
  }});
}}

function handleSubmit() {{
  submitted=true;
  const ans=Object.keys(answers).length;
  const pct=total>0?Math.round(ans/total*100):0;
  const footer=document.getElementById('footer');
  footer.innerHTML='';
  const sc=document.createElement('div'); sc.className='score-card';
  sc.innerHTML='<div class="score-big">'+ans+' / '+total+'</div>'
    +'<div class="score-sub">questions answered</div>'
    +'<div class="score-pct">'+pct+'% completion</div>';
  footer.appendChild(sc);
  const rb=document.createElement('button'); rb.className='reset-btn';
  rb.textContent='↺  Start over';
  rb.onclick=()=>{{
    answers={{}};submitted=false;renderAll();updateProgress();
    const f=document.getElementById('footer'); f.innerHTML='';
    const sb=document.createElement('button');
    sb.className='submit-btn';sb.id='submitBtn';sb.disabled=true;
    sb.textContent='Submit answers';sb.onclick=handleSubmit;
    f.appendChild(sb);updateProgress();
  }};
  footer.appendChild(rb);
  renderAll(); updateProgress();
}}

renderAll(); updateProgress();
</script>
</body>
</html>"""

# ── TABS ──────────────────────────────────────────────────────
tab_gen, tab_dig = st.tabs(["🧠 Generate Quiz", "📄 Digitize Exam PDF"])

# ── TAB 1 ─────────────────────────────────────────────────────
with tab_gen:
    with st.sidebar:
        st.title("⚙️ PFE Admin Console")
        pdf  = st.file_uploader("Upload Course PDF", type="pdf", key="gen_pdf")
        mode = st.selectbox("Exam Mode", ["MultiChoice","Essay","ShortAnswer","Mixed Mode"])
        if mode == "Mixed Mode":
            with st.form("mix_form"):
                c_mcq  = st.number_input("MultiChoice", 0, 10, 2)
                c_ess  = st.number_input("Essay",       0, 10, 1)
                c_shrt = st.number_input("ShortAnswer", 0, 10, 1)
                gen    = st.form_submit_button("Generate")
                req    = f"EXACTLY {c_mcq} MultiChoice, {c_ess} Essay, {c_shrt} ShortAnswer"
        else:
            q_count = st.slider("Questions", 1, 10, 3)
            gen     = st.button(f"Generate {mode} Quiz")
            req     = f"EXACTLY {q_count} questions of type '{mode}'"

        if pdf and gen:
            with st.spinner("Generating..."):
                txt, imgs = process_pdf(pdf.read())
                st.session_state.visual_assets = imgs
                prompt = f"""[INST]
SYSTEM: Strict academic examiner.
TASK: Generate a quiz. REQUIRED: {req}
RULES: 1.MultiChoice needs 4 options. 2.Essay needs sampleAnswer. 3.context=exact quote.
SOURCE: {txt[:5000]}
[/INST]"""
                raw = quiz_gen.invoke(prompt)
                if mode != "Mixed Mode":
                    for q in raw.questions: q.type = mode
                st.session_state.quiz_data = raw

    if st.session_state.visual_assets:
        with st.expander("🖼️ Visuals"):
            cols = st.columns(3)
            for i, img in enumerate(st.session_state.visual_assets[:6]):
                cols[i%3].image(img, use_container_width=True)

    render_quiz(st.session_state.quiz_data, key_prefix="gen")
    with st.expander("🛠️ Raw Output"):
        if st.session_state.quiz_data:
            st.json(st.session_state.quiz_data.dict())

# ── TAB 2 ─────────────────────────────────────────────────────
with tab_dig:
    st.header("📄 Digitize an Existing Exam")
    st.write(
        "Upload an exam PDF. Pages render **pixel-perfect**. "
        "MCQ questions show clickable A/B/C/D buttons. "
        "Diagram questions show a **View in PDF** button that scrolls and highlights the figure. "
        "Essay questions show a text box."
    )

    dig_pdf = st.file_uploader("Upload Exam PDF", type="pdf", key="dig_pdf")

    if dig_pdf:
        if st.button("🔍 Digitize Exam"):
            pdf_bytes = dig_pdf.read()

            with st.spinner("Rendering pages..."):
                st.session_state.digitized_pages = pdf_to_b64_pages(pdf_bytes, dpi=150)

            with st.spinner("Extracting questions..."):
                st.session_state.digitized_questions = parse_cambridge_pdf(pdf_bytes)
                st.session_state.figure_coords       = extract_figure_coords(pdf_bytes, dpi=150)

            n     = len(st.session_state.digitized_questions)
            mcqn  = sum(1 for q in st.session_state.digitized_questions if q["type"]=="MCQ")
            essn  = sum(1 for q in st.session_state.digitized_questions if q["type"]=="Essay")
            diagn = sum(1 for q in st.session_state.digitized_questions if q["type"]=="Diagram")
            st.success(f"✅ {n} questions — {mcqn} MCQ · {essn} Essay · {diagn} Diagram/Table")

    if st.session_state.digitized_pages and st.session_state.digitized_questions:
        html = build_digitizer_html(
            st.session_state.digitized_pages,
            st.session_state.digitized_questions,
            st.session_state.get("figure_coords", {}),
        )
        components.html(html, height=900, scrolling=False)

    with st.expander("🛠️ Debug — extracted questions"):
        if st.session_state.digitized_questions:
            st.json(st.session_state.digitized_questions)
          
def api_digitize(pdf_bytes):
    questions = parse_cambridge_pdf(pdf_bytes)
    return {
        "count": len(questions),
        "questions": questions
    }
def api_generate(text, mode="mixed", mcq=5, essay=2):
    prompt = f"""
    Generate a quiz from this content:

    {text[:5000]}
    """

    result = quiz_gen.invoke(prompt)

    return result.dict()

def api_grade(question, answer):
    prompt = f"""
    Question: {question}
    Student answer: {answer}

    Give score and feedback.
    """

    result = grader.invoke(prompt)

    return result.dict()
