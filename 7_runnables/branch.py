
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnableBranch, RunnablePassthrough

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

parser = StrOutputParser()

prompt1 = PromptTemplate(
    template= 'Generate a news on {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template= 'Summarize above topic in 2 lines. {topic}' ,
    input_variables= ['topic']
)

news_chain = prompt1 | model | parser

branch_chain = RunnableBranch(
    (lambda x:len(x.split()) > 50, prompt2 | model | parser), # if condition
    RunnablePassthrough()
)

final_chain = news_chain | branch_chain

result = final_chain.invoke({'topic': 'Over use of social media.'})

print(result)