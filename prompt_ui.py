from langchain_google_genai import ChatGoogleGenerativeAI
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model='gemini-2.5-flash')

st.header("Research Assistant Tool")

user_input = st.text_input("Enter your prompt:")

if st.button("Summarize"):
    result = model.invoke(user_input) #static prompt
    st.write(result.content)