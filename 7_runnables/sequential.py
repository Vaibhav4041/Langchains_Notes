
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence
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
    template = 'Write a song on {topic}',
    input_variables=['topic']
)

parser = StrOutputParser()

# chain1 = RunnableSequence(prompt, model, parser)

prompt2 = PromptTemplate(
    template= 'Explain me the following content {text}',
    input_variables=['text']
)

# chain2 = RunnableSequence(prompt2, model, parser)

seq_chain = RunnableSequence(prompt, model, parser, prompt2, model, parser)

print(seq_chain.invoke('IT lifestyle'))