import os
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from typing import TypedDict
load_dotenv()


llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.2-3B-Instruct", # Use this stable model instead
    task="text-generation",
    max_new_tokens=512,
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
)

model = ChatHuggingFace(llm=llm)

# schema
class Review(TypedDict):
    summary: str
    sentiment: str

strectured_model = model.with_structured_output(Review)

result = strectured_model.invoke("""The movie was fantastic! 
I loved the plot and the characters were very well developed.
Also, the cinematography was stunning.""")


print(result)
print(result['summary']
      , result['sentiment'])
