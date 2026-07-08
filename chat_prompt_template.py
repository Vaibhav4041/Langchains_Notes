from langchai_core.prompts import ChatPromptTemplate
from langchain_core.mesages import HumanMessage, AIMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
import streamlit as st
from dotenv import load_dotenv
load_dotenv()
model = ChatGoogleGenerativeAI(model='gemini-2.5-flash')

# ChatPromptTemplate.from_messages same as below
chat_template = ChatPromptTemplate(  [
    ("system", "You are a helpful assistant who explains {role} concepts in detail." ),
    ("human", "Explain the concept of {topic}" )
    ] )

prompt = chat_template.invoke({
    "role": "cricket",
    "topic": "Dot ball" 
})