from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
import os
load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.2-3B-Instruct",
    task="text-generation",
    max_new_tokens=1024, # Increased for structured formatting
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
)


model = ChatHuggingFace(llm=llm)

# 1st Prompt Detailed Prompt
template1 = PromptTemplate(
    template="Write a detailed report on {topic}",
    input_variables=["topic"]
)

# 2nd Prompt Summary

template2 = PromptTemplate(
    template="Write a 5 line summary on the following text. /n {text}",
    input_variables=["text"]
)

prompt1 = template1.invoke({"topic": "The impact of climate change on global agriculture."})

result1 = model.invoke(prompt1)

prompt2 = template2.invoke({"text": result1.content})

result2 = model.invoke(prompt2)
print("Detailed Report:\n", result2.content)