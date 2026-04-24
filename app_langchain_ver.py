import streamlit as st
import json
from PyPDF2 import PdfReader
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field, AliasChoices
from typing import List, Optional

# --- 1. SCHEMA ---
class Option(BaseModel):
    text: str = Field(validation_alias=AliasChoices("text", "option_text"))
    isCorrect: bool = Field(default=False, validation_alias=AliasChoices("isCorrect", "is_correct"))

class Question(BaseModel):
    questionText: str = Field(validation_alias=AliasChoices("questionText", "text"))
    type: str = Field(default="MultiChoice")
    options: Optional[List[Option]] = None
    sampleAnswer: Optional[str] = None
    context: str = Field(default="Source not cited.")

class Quiz(BaseModel):
    title: str = Field(validation_alias=AliasChoices("title", "quiz_name"))
    questions: List[Question]

# --- 2. CONFIG ---
st.set_page_config(layout="wide", page_title="AI Pedagogical Platform")
llm = OllamaLLM(model="gemma3:4b", base_url="http://localhost:11434", temperature=0.1, num_predict=4096)
base_parser = PydanticOutputParser(pydantic_object=Quiz)

if "quiz_data" not in st.session_state: st.session_state.quiz_data = None

# --- 3. THE REPAIR ENGINE ---
def robust_parse(completion):
    """Manually cleans the AI output before Pydantic sees it."""
    try:
        # Extract JSON if the model included conversational filler
        json_str = re.search(r"\{.*\}", completion, re.DOTALL).group()
        data = json.loads(json_str)
        
        # FIX: If 'options' are strings, convert them to objects
        for q in data.get("questions", []):
            if "options" in q and isinstance(q["options"], list):
                new_opts = []
                for opt in q["options"]:
                    if isinstance(opt, str):
                        # Assume the first option generated is the correct one if AI forgot booleans
                        new_opts.append({"text": opt, "isCorrect": "result" in opt.lower() or "correct" in opt.lower()})
                    else:
                        new_opts.append(opt)
                q["options"] = new_opts
        
        return base_parser.parse(json.dumps(data))
    except Exception:
        # Fallback to standard parser if manual clean fails
        return base_parser.parse(completion)

def process_document(pdf_file, q_type, count):
    reader = PdfReader(pdf_file)
    raw_text = " ".join([p.extract_text() for p in reader.pages if p.extract_text()])
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=3500, chunk_overlap=200)
    docs = text_splitter.split_text(raw_text)

    prompt = PromptTemplate(
        template="""[INST] <<SYS>>
        You are an academic examiner. 
        MANDATORY: For MultiChoice, options MUST be objects: {{"text": "...", "isCorrect": true/false}}.
        {format_instructions}
        <</SYS>>
        Generate {count} {q_type} questions based on: {text} [/INST]""",
        input_variables=["count", "q_type", "text"],
        partial_variables={"format_instructions": base_parser.get_format_instructions()}
    )

    # We call the LLM directly so we can clean the string before parsing
    raw_response = llm.invoke(prompt.format(count=count, q_type=q_type, text=docs[0]))
    return robust_parse(raw_response)

import re # Needed for the regex in robust_parse

# --- 4. SIDEBAR ---
with st.sidebar:
    st.title("PFE Admin Console")
    q_format = st.selectbox("Format", ["MultiChoice", "ShortAnswer", "Essay"])
    q_count = st.slider("Questions", 1, 10, 5) 
    pdf = st.file_uploader("Upload PDF", type="pdf")
    
    if pdf and st.button("Generate Quiz"):
        with st.spinner("Processing..."):
            try:
                st.session_state.quiz_data = process_document(pdf, q_format, q_count)
            except Exception as e:
                st.error(f"Logic Error: {e}")

# --- 5. INTERFACE ---
if st.session_state.quiz_data:
    quiz = st.session_state.quiz_data
    st.header(f"📝 {quiz.title}")
    
    for i, q in enumerate(quiz.questions):
        with st.container(border=True):
            st.markdown(f"**Q{i+1}: {q.questionText}**")
            
            # Use the format selected in sidebar to decide the UI
            if q_format == "MultiChoice" and q.options:
                choices = [o.text for o in q.options]
                ans = st.radio("Select one:", choices, key=f"r{i}", index=None)
                if ans:
                    correct = next((o.text for o in q.options if o.isCorrect), choices[0])
                    if ans == correct: st.success("Correct!")
                    else: st.error(f"Incorrect. Answer: {correct}")
            else:
                st.text_area("Answer:", key=f"t{i}")
                if st.button("Check Reference", key=f"b{i}"):
                    st.info(f"**Reference:** {q.sampleAnswer or 'See context.'}")
                    st.caption(f"Evidence: {q.context}")

    with st.expander("🛠️ Raw Data"):
        st.json(quiz.dict())
