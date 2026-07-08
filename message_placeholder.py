from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
import streamlit as st
from dotenv import load_dotenv
load_dotenv()

model = ChatGoogleGenerativeAI(model='gemini-2.5-flash')

# Chat template
chat_template = ChatPromptTemplate([
    ('system', 'You are a helpful customer support agent.'),
    MessagesPlaceholder(variable_name='chat_history'),
    ('human', '{query}')
])

# load chat history 

chat_history = []

with open('chat_history.txt') as f:
    chat_history.extend(f.readlines())

#print(chat_history)
#  chat prompt
prompt = chat_template.invoke({'chat_history': chat_history, 'query':' Where is my refund ?'})
#print(prompt)


while True:
    user_input = input("You: ")
    prompt = chat_template.invoke({'chat_history': chat_history, 'query':user_input})
    chat_history.append(HumanMessage(content=user_input))
    if user_input == 'exit':
        break
    result = model.invoke(prompt)
    chat_history.append(AIMessage(content=result.content))
    print(result.content)
print(chat_history)