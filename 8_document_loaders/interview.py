import streamlit as st
import os
import time
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --- INITIALIZATION ---
load_dotenv()
st.set_page_config(page_title="Elite Interviewer AI", page_icon="🎓", layout="wide")

# Modern UI Styling
st.markdown("""
    <style>
    .stProgress > div > div > div > div { background-color: #00FFAA; }
    .question-card { background: #ffffff; padding: 20px; border-radius: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 20px; border-left: 5px solid #6C63FF; }
    .metric-container { background: #f0f2f6; padding: 15px; border-radius: 10px; text-align: center; }
    </style>
    """, unsafe_allow_html=True)

# --- CORE LLM LOGIC ---
@st.cache_resource
def load_llm():
    return HuggingFaceEndpoint(
        repo_id="meta-llama/Llama-3.2-3B-Instruct",
        task="chat-completion",
        temperature=0.6,
        max_new_tokens=1024,
        huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
    )

llm_engine = load_llm()
model = ChatHuggingFace(llm=llm_engine)

# --- SESSION STATE TRACKING ---
if "step" not in st.session_state:
    st.session_state.update({
        "step": "UPLOAD", # UPLOAD -> INTERVIEW -> RESULTS
        "round": 1,
        "questions": [],
        "user_answers": {},
        "notes": "",
        "total_score": 0,
        "history": []
    })

# --- FUNCTIONS ---
def get_pdf_content():
    if not os.path.exists('pdfs') or not os.listdir('pdfs'):
        return None
    loader = DirectoryLoader('pdfs', glob="*.pdf", loader_cls=PyPDFLoader)
    docs = loader.load()
    return " ".join([d.page_content for d in docs])

def fetch_questions(round_num, notes):
    levels = {1: "Fundamental/Basic", 2: "Intermediate/Analytical", 3: "Advanced/Hard/Architectural"}
    template = """You are a Lead Technical Interviewer. Based on these study notes:
    {notes}
    
    Generate exactly 5 {level} questions. 
    Format: Return ONLY the questions separated by double newlines. No intro text."""
    
    prompt = PromptTemplate.from_template(template)
    chain = prompt | model | StrOutputParser()
    # Use first 6000 chars to avoid context overflow
    response = chain.invoke({"level": levels[round_num], "notes": notes[:6000]})
    return [q.strip() for q in response.split('\n\n') if q.strip()][:5]

def analyze_answers(answers_dict, notes):
    template = """Analyze these interview responses against the source notes.
    Notes: {notes}
    Responses: {responses}
    
    Provide:
    1. A score out of 100.
    2. Strengths.
    3. Weaknesses.
    4. Ideal answers for missed points.
    Format as clean Markdown."""
    
    prompt = PromptTemplate.from_template(template)
    chain = prompt | model | StrOutputParser()
    return chain.invoke({"notes": notes[:5000], "responses": str(answers_dict)})

# --- USER INTERFACE ---

# Sidebar Progress
with st.sidebar:
    st.title("🏆 Prep Dashboard")
    if st.session_state.step != "UPLOAD":
        progress_val = (st.session_state.round - 1) / 3
        st.write(f"**Overall Progress**")
        st.progress(progress_val)
        st.metric("Current Round", f"{st.session_state.round} / 3")
    
    if st.button("🔄 Reset Session"):
        st.session_state.clear()
        st.rerun()

# Main App Flow
if st.session_state.step == "UPLOAD":
    st.title("🚀 AI Interview Prep: Phase 1")
    st.info("Place your PDF notes in the `/pdfs` folder of this project.")
    
    if st.button("Start My Interview", use_container_width=True):
        with st.spinner("Processing your knowledge base..."):
            content = get_pdf_content()
            if content:
                st.session_state.notes = content
                st.session_state.questions = fetch_questions(1, content)
                st.session_state.step = "INTERVIEW"
                st.rerun()
            else:
                st.error("No PDFs found in the /pdfs folder!")

elif st.session_state.step == "INTERVIEW":
    round_titles = {1: "🟢 Round 1: The Basics", 2: "🟡 Round 2: Depth & Logic", 3: "🔴 Round 3: Mastery"}
    st.title(round_titles[st.session_state.round])
    
    with st.form(f"round_form_{st.session_state.round}"):
        current_answers = []
        for i, q in enumerate(st.session_state.questions):
            st.markdown(f"<div class='question-card'><b>Question {i+1}:</b><br>{q}</div>", unsafe_allow_html=True)
            ans = st.text_area("Your Response", key=f"ans_{st.session_state.round}_{i}", placeholder="Type your detailed answer here...")
            current_answers.append({"q": q, "a": ans})
        
        btn_label = "Proceed to Next Round" if st.session_state.round < 3 else "Finish & Analyze"
        if st.form_submit_button(btn_label, use_container_width=True):
            st.session_state.user_answers[st.session_state.round] = current_answers
            
            if st.session_state.round < 3:
                st.session_state.round += 1
                with st.spinner("Generating higher-level questions..."):
                    st.session_state.questions = fetch_questions(st.session_state.round, st.session_state.notes)
                st.rerun()
            else:
                st.session_state.step = "RESULTS"
                st.rerun()

elif st.session_state.step == "RESULTS":
    st.title("📊 Final Performance Report")
    
    with st.spinner("Generating deep analysis..."):
        report = analyze_answers(st.session_state.user_answers, st.session_state.notes)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### Quick Stats")
        st.success("✅ 15/15 Questions Answered")
        st.info("💡 Focus Area: High-Level Concepts")
    
    with col2:
        st.markdown("### Detailed Feedback")
        st.markdown(report)
    
    if st.button("Download Feedback (Text)"):
        st.download_button("Click to Download", report, "interview_feedback.txt")