
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableParallel
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


prompt1 = PromptTemplate(
    template = 'Generate a paper based News on {topic}',
    input_variables= ['topic']
)

prompt2 = PromptTemplate(
    template = 'Generate a news for digital blog on {topic}',
    input_variables = ['topic'] 
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'paper_news': RunnableSequence(prompt1, model, parser) ,
    'blog_news': RunnableSequence(prompt2, model, parser)
})


result = parallel_chain.invoke('India won worldcup in 2011')

print(result)


print('_' * 80)


print(result['paper_news'])

print('_' * 80)

print(result['blog_news'])