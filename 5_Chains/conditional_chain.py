from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field 
from langchain.schema.runnable import RunnableParallel, RunnableBranch, RunnableLambda
from typing import Literal
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

class Feedback(BaseModel):
    sentiment: Literal["positive", "negative"] = Field(
        description="Sentiment of the feedback"
    )

parser = PydanticOutputParser(pydantic_object=Feedback)

prompt = PromptTemplate(
    template="""
You are a sentiment classifier.

Classify the following feedback as either "positive" or "negative".

Feedback:
{feedback}

{format_instruction}

Return ONLY valid JSON.
""",
    input_variables=["feedback"],
    partial_variables={"format_instruction": parser.get_format_instructions()},
)

chain = prompt | model | parser

# result = chain.invoke({"feedback": "This is the best phone ever."})
# print(result)

# Branch chain

branch_chain = R