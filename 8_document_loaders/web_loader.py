from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader, PyPDFLoader, WebBaseLoader
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
    template='Answer the following question \n {question} from the following web data \n {webdata}' ,
    input_variables=['question', 'webdata']
)

parser = StrOutputParser()

url = 'https://www.amazon.in/Sony-PlayStation%C2%AE5-Digital-Edition-slim/dp/B0CY5QW186/ref=s9_acsd_al_ot_cv2_0_t?_encoding=UTF8&pf_rd_m=A21TJRUUN4KGV&pf_rd_s=merchandised-search-5&pf_rd_r=5P5M029CE8GTVHE3AR49&pf_rd_p=904bd1a4-ede9-4776-9ddb-0893da74c3ea&pf_rd_t=976460031&pf_rd_i=21725163031'

loader = WebBaseLoader(url)

docs = loader.load()

chain = prompt | model | parser

print(chain.invoke({'question': 'This product comes with any pre installed games init ?', 'webdata':docs}))