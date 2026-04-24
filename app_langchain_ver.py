import streamlit as st
import json
import re
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
    # REMOVED the default "Source not cited" to force the AI to provide real data
    context: str = Field(validation_alias=AliasChoices("context", "source_quote", "evidence"))

class Quiz(BaseModel):
    title: str = Field(validation_alias=AliasChoices("title", "quiz_name"))
    questions: List[Question]

# --- 2. CONFIG ---
st.set_page_config(layout="wide", page_title="AI Pedagogical Platform")
llm = OllamaLLM(model="gemma3:4b", base_url="http://localhost:11434", temperature=0.1, num_predict=4096)
base_parser = PydanticOutputParser(pydantic_object=Quiz)

if "quiz_data" not in st.session_state: st.session_state.quiz_data = None

# --- 3. THE REPAIR ENGINE (With Evidence-Recovery) ---
def robust_parse(completion):
    try:
        json_str = re.search(r"\{.*\}", completion, re.DOTALL).group()
        data = json.loads(json_str)
        
        for q in data.get("questions", []):
            # MCQ FIX
            if "options" in q and isinstance(q["options"], list):
                new_opts = []
                for opt in q["options"]:
                    if isinstance(opt, str):
                        new_opts.append({"text": opt, "isCorrect": "correct" in opt.lower()})
                    else:
                        new_opts.append(opt)
                q["options"] = new_opts
            
            # EVIDENCE RECOVERY FIX:
            # If the AI forgot 'context' but put a quote in 'sampleAnswer', we swap them.
            if not q.get("context") or q.get("context") == "Source not cited.":
                if q.get("sampleAnswer") and len(q["sampleAnswer"]) > 20:
                    q["context"] = q["sampleAnswer"]
                else:
                    q["context"] = "Refer to the uploaded document for this answer."
        
        return base_parser.parse(json.dumps(data))
    except Exception:
        return base_parser.parse(completion)

def process_document(pdf_file, q_type, count):
    reader = PdfReader(pdf_file)
    raw_text = " ".join([p.extract_text() for p in reader.pages if p.extract_text()])
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=3500, chunk_overlap=200)
    docs = text_splitter.split_text(raw_text)

    # We use very aggressive wording here to make the AI scared to skip the context
    prompt = PromptTemplate(
        template="""[INST] <<SYS>>
        You are a meticulous academic examiner. 
        MANDATORY: You MUST fill the 'context' field with the EXACT sentence from the text that proves the answer. 
        NEVER leave the 'context' field empty.
        {format_instructions}
        <</SYS>>
        Generate {count} {q_type} questions based on: {text} [/INST]""",
        input_variables=["count", "q_type", "text"],
        partial_variables={"format_instructions": base_parser.get_format_instructions()}
    )

    raw_response = llm.invoke(prompt.format(count=count, q_type=q_type, text=docs[0]))
    return robust_parse(raw_response)

import re

# --- 4. SIDEBAR ---
with st.sidebar:
    st.title("PFE Admin Console")
    q_format = st.selectbox("Format", ["MultiChoice", "ShortAnswer", "Essay"])
    q_count = st.slider("Questions", 1, 10, 5) 
    pdf = st.file_uploader("Upload PDF", type="pdf")
    
    if pdf and st.button("Generate Quiz"):
        with st.spinner("Analyzing and extracting quotes..."):
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
            
            if q_format == "MultiChoice" and q.options:
                choices = [o.text for o in q.options]
                ans = st.radio("Select one:", choices, key=f"r{i}", index=None)
                if ans:
                    correct = next((o.text for o in q.options if o.isCorrect), choices[0])
                    if ans == correct: st.success("Correct!")
                    else: st.error(f"Incorrect. Answer: {correct}")
            else:
                st.text_area("Answer:", key=f"t{i}")
                if st.button("Verify & View Source", key=f"b{i}"):
                    # We show both the model answer AND the source evidence
                    st.info(f"**Reference:** {q.sampleAnswer or 'Verify with the quote below.'}")
                    st.success(f"**📍 Source Evidence from PDF:**\n\n{q.context}")

    with st.expander("🛠️ Raw Data"):
        st.json(quiz.dict())
