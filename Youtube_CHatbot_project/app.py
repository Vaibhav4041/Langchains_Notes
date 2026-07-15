from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFaceEmbeddings
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader, PyPDFLoader, DirectoryLoader
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

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",
    model_kwargs={'device': 'cpu'} # Change to 'cuda' if you have a GPU
)

# Example usage:
text = "This is a test document."
query_result = embeddings.embed_query(text)

print(f"Embedding length: {len(query_result)}")
print(query_result)

model = ChatHuggingFace(llm=llm)