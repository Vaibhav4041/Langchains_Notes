from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
import os

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.2-3B-Instruct",
    task="chat-completion",
    temperature=0.0,
    max_new_tokens=300,
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
)

model = ChatHuggingFace(llm=llm)

prompt = PromptTemplate(
    template='Based on above my notes ask me any ideal 5 interview questions - \n {notes}' ,
    input_variables=['notes']
)

parser = StrOutputParser()

loader = PyPDFLoader('./pdfs/langchain_notes1.pdf')

docs = loader.load()

# print(docs[47].page_content) #uses one page based docs for each page

# print(docs[0].metadata)

chain = prompt | model | parser

print(chain.invoke({'notes':docs}))