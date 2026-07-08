from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate,load_prompt

import streamlit as st
from dotenv import load_dotenv
# We are going to use readymade template for prompt by importing PromptTemplate from langchain_core.prompts
load_dotenv()

model = ChatGoogleGenerativeAI(model='gemini-2.5-flash')

st.header("Research Assistant Tool")

paper_input = st.selectbox( "Select Research Paper Name", ["Select...", "Attention Is All You Need",
"BERT: Pre-training of Deep Bidirectional Transformers", "GPT-3: Language Models are Few-Shot Learners", "Diffusion Models Beat GANs on Image Synthesis"] )

style_input = st.selectbox( "Select Explanation Style", ["Beginner-Friendly", "Technical", "CodeOriented", "Mathematical"] )

length_input = st.selectbox( "Select Explanation Length", ["Short (1-2  paragraphs)", "Medium (3-5paragraphs)", "Long (detailed explanation)"] )

template = load_prompt('research_summary_template.json')

# fill placeholders with user inputs
prompt = template.invoke({
    "paper_input": paper_input,
    "style_input": style_input,
    "length_input": length_input
})

if st.button("Summarize"):
    result = model.invoke(prompt) #dynamic prompt
    st.write(result.content)