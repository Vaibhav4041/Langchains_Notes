#annotations using TypedDict
import os
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from typing import Annotated, TypedDict, Optional
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
    key_themes: Annotated[list[str],'Write down all the key themes discussed in the review in list']
    summary: Annotated[str, "A brief summary of the review" ]
    sentiment: Annotated[str, "Returns the sentiment of the review: positive, negative, or neutral" ]
    pros:Annotated[Optional[list[str]], "List the pros mentioned in the review, if any" ]
    cons:Annotated[Optional[list[str]], "List the cons mentioned in the review, if any" ]
    name : Annotated[Optional[str], "Name of the reviewer, if mentioned" ]
strectured_model = model.with_structured_output(Review)

result = strectured_model.invoke(''' I m a long-time patient of this hospital; 
every time I come in, I have to wait over an hour to get seen. 
The staff is nice, and the new renovations look good, 
but I didn't leave work early to wait an hour to be seen
Review by Vaibhav Bhosale  ''')


print(result)
