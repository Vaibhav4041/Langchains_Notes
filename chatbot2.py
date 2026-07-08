# langchain having 3 types of messages - HumanMessage, AIMessage, SystemMessage
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
import streamlit as st
from dotenv import load_dotenv
load_dotenv()

model = ChatGoogleGenerativeAI(model='gemini-2.5-flash')

chat_history =[
    SystemMessage(content="You are a helpful Assistant.")
]

while True:
    user_input = input("You: ")
    chat_history.append(HumanMessage(content=user_input))
    if user_input == 'exit':
        break
    result = model.invoke(chat_history)
    chat_history.append(AIMessage(content=result.content))
    print(result.content)
print(chat_history)