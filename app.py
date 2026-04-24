import streamlit as st
import json
import re
from openai import OpenAI
import PyPDF2

# --- CONFIGURATION ---
st.set_page_config(layout="wide", page_title="Stress Tester")
client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

# --- UTILS: THE REPAIR MAN ---
def repair_json(json_string):
    """Automatically closes missing brackets if the AI cuts off due to PC memory limits."""
    brackets = []
    for char in json_string:
        if char == '{': brackets.append('}')
        elif char == '[': brackets.append(']')
        elif char == '}': 
            if brackets and brackets[-1] == '}': brackets.pop()
        elif char == ']':
            if brackets and brackets[-1] == ']': brackets.pop()
    return json_string + "".join(reversed(brackets))

if "quiz_data" not in st.session_state: st.session_state.quiz_data = None
if "debug_content" not in st.session_state: st.session_state.debug_content = None

# --- PROMPT ENGINE ---
def get_quiz_prompt(text, q_type, count):
    return f"""
[INST] <<SYS>>
Output ONLY valid raw JSON. No markdown. No talking.
<</SYS>>
Generate EXACTLY {count} {q_type} questions. 
Distribute questions across the Intro, Middle, and End of this text.

TEXT:
{text[:8000]}

STRUCTURE:
{{
  "title": "Technical Quiz",
  "quiz": {{
    "questions": [
      {{
        "questionText": "### Question",
        "type": "{q_type}",
        "options": [{{"text": "A", "isCorrect": true}}, {{"text": "B", "isCorrect": false}}],
        "sampleAnswer": "Required for ShortAnswer/Essay",
        "context": "Source quote"
      }}
    ]
  }}
}}
[/INST]
"""

# --- SIDEBAR ---
with st.sidebar:
    st.title("Stress Test Panel")
    selected_type = st.selectbox("Format", ["MultiChoice", "ShortAnswer", "Essay"])
    selected_count = st.slider("Quantity", 1, 10, 8) 
    pdf_file = st.file_uploader("Upload PDF", type="pdf")
    
    if pdf_file and st.button("Run Generation"):
        try:
            reader = PyPDF2.PdfReader(pdf_file)
            if reader.is_encrypted: reader.decrypt("")
            raw_text = " ".join([p.extract_text() for p in reader.pages if p.extract_text()])
            
            with st.spinner("AI is working..."):
                response = client.chat.completions.create(
                    model="gemma3:4b",
                    messages=[{"role": "user", "content": get_quiz_prompt(raw_text, selected_type, selected_count)}],
                    temperature=0.1
                )
                
                full_content = response.choices[0].message.content
                st.session_state.debug_content = full_content
                json_match = re.search(r'(\{.*\}|\[.*\])', full_content, re.DOTALL)
                
                if json_match:
                    repaired = repair_json(json_match.group(0))
                    parsed_data = json.loads(repaired)
                    
                    if isinstance(parsed_data, list):
                        parsed_data = {"quiz": {"questions": parsed_data}}
                    elif 'questions' in parsed_data and 'quiz' not in parsed_data:
                        parsed_data = {"quiz": {"questions": parsed_data['questions']}}
                    
                    st.session_state.quiz_data = parsed_data
                else:
                    st.error("No JSON found.")
        except Exception as e:
            st.error(f"Error: {e}")

# --- MAIN UI ---
if st.session_state.quiz_data:
    data = st.session_state.quiz_data
    tab_q, tab_j = st.tabs(["Quiz", "JSON"])
    
    with tab_q:
        st.title(data.get('title', 'Technical Quiz'))
        
        questions = data.get('quiz', {}).get('questions', []) if isinstance(data, dict) else []
        
        for i, q in enumerate(questions):
            st.divider()
            st.markdown(q.get('questionText', 'Question'))
            
            # Normalize type for comparison (converts 'ShortAnswer' to 'shortanswer')
            current_type = str(q.get('type', '')).lower()

            if current_type == "multichoice":
                opts = [o['text'] for o in q.get('options', [])]
                ans = st.radio("Select:", opts, key=f"q_{i}", index=None)
                if ans:
                    correct = next((o['text'] for o in q['options'] if o['isCorrect']), "N/A")
                    if ans == correct:
                        st.success("Correct")
                    else:
                        st.error(f"Wrong: {correct}")
                    with st.expander("Show Context"):
                        st.write(q.get('context'))
            
            # Integrated check for both ShortAnswer and Essay types
            elif current_type in ["shortanswer", "essay"]:
                st.text_area("Answer:", key=f"a_{i}")
                if st.button("Evaluate", key=f"b_{i}"):
                    st.markdown("#### Reference Answer")
                    st.write(q.get('sampleAnswer'))
                    with st.expander("View Source Context"):
                        st.write(q.get('context'))

    with tab_j:
        st.json(data)

    if st.button("Clear"):
        st.session_state.quiz_data = None
        st.rerun()

elif st.session_state.debug_content:
    with st.expander("Debug Output"):
        st.code(st.session_state.debug_content)