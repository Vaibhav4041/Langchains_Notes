import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

load_dotenv()

# 1. Setup the underlying Inference Endpoint
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.2-3B-Instruct", # Use this stable model instead
    task="text-generation",
    max_new_tokens=512,
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
)


# 2. Wrap it in ChatHuggingFace to handle the System/Human/AI message logic correctly
model = ChatHuggingFace(llm=llm)

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


while True:
    user_input = input("User: ")
    prompt = chat_template.invoke({'chat_history': chat_history, 'query':user_input})
    chat_history.append(HumanMessage(content=user_input))
    if user_input == 'exit':
        break
    result = model.invoke(prompt)
    chat_history.append(AIMessage(content=result.content))
    print(result.content)
print(chat_history)

print("Llama-3 Ready! (Type 'exit' to stop)")

while True:
    user_input = input("User: ")
    if user_input.lower() == 'exit':
        break
    
    # 1. Create the prompt including history
    prompt = chat_template.invoke({'chat_history': chat_history, 'query': user_input})
    
    # 2. Get response from Gemma 3
    result = model.invoke(prompt)
    
    # 3. Update history
    chat_history.append(HumanMessage(content=user_input))
    chat_history.append(AIMessage(content=result.content))
    
    print(f"AI: {result.content}")
