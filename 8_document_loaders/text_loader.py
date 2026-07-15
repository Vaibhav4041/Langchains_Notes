from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
import os

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.2-3B-Instruct",
    task="chat-completion",
    temperature=0.0,
    max_new_tokens=128,
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
)

model = ChatHuggingFace(llm=llm)

prompt = PromptTemplate(
    template='Write a summary for the following article - \n {article}' ,
    input_variables=['article']
)

parser = StrOutputParser()

loader = TextLoader('agentic_ai_rev.txt', encoding = 'utf-8')

docs = loader.load()

# print(docs)  # showcasing meata and page content

#print(docs[0].page_content)

#print(docs[0].metadata)

chain = prompt | model | parser

print(chain.invoke({'article':docs[0].page_content}))