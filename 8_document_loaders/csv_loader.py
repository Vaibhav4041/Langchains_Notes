from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader, PyPDFLoader, DirectoryLoader, CSVLoader
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
    template='Based on above csv data tell me which is the maximum EstimatedSalary salary having there - \n {data}' ,
    input_variables=['data']
)

parser = StrOutputParser()

loader = CSVLoader(file_path = 'Social_Network_Ads.csv')

docs = loader.load()

# print(docs[1]) creates new doc for each row 

chain = prompt | model | parser

print(chain.invoke({'data':docs}))