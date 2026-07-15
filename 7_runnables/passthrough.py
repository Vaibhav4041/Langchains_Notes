
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough
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
    template='Generate a song on {topic}',
    input_variables= ['topic']
)

prompt2 = PromptTemplate(
    template='Explain me above song in simple term {topic}' ,
    input_variables=['topic']
)

song_chain = prompt1 | model | parser

par_chain = RunnableParallel({
    'song': RunnablePassthrough() ,
    'explain': prompt2 | model | parser
})


final_CHAIN = song_chain | par_chain

result = final_CHAIN.invoke({'topic':'India'})

print(result)

print(result['song'])

print(result['explain'])