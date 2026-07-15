
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnableBranch, RunnableLambda, RunnablePassthrough

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

def word_counter(text):
    return len(text.split())


#chain = RunnableLambda(word_counter)

#print(chain.invoke('Hello Agentic AI world'))

prompt1 = PromptTemplate(
    template = 'Generate a news on {topic}' ,
    input_variables=['topic']
)

news_gen_chain = RunnableSequence(prompt1, model, parser)

par_chain = RunnableParallel({
    'news': RunnablePassthrough(),
    'word_count': RunnableLambda(word_counter)
})

final_chain = RunnableSequence(news_gen_chain, par_chain)

result = final_chain.invoke('9/11 plane crash incident')

print(result)

print('*' * 80)

print(result['news'])

print('*' * 80)

print(result['word_count'])