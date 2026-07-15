import os
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Optional, Literal

load_dotenv()

# 1. Initialize the LLM
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.2-3B-Instruct",
    task="text-generation",
    max_new_tokens=1024, # Increased for structured formatting
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
)

model = ChatHuggingFace(llm=llm)

# 2. Define Schema (Same as yours)
class Review(BaseModel):
    key_themes: list[str] = Field(description='Key themes discussed in the review')
    summary: str = Field(description="A brief summary of the review")
    sentiment: Literal['pos', 'neg'] = Field(description="Sentiment: pos or neg")
    pros: Optional[list[str]] = Field(default=None)
    cons: Optional[list[str]] = Field(default=None)
    name: Optional[str] = Field(default=None)

# 3. Setup Parser and Prompt
parser = PydanticOutputParser(pydantic_object=Review)

# The prompt must explicitly ask for the JSON format
prompt_template = PromptTemplate(
    template="Analyze the following review.\n{format_instructions}\nReview: {review}\n",
    input_variables=["review"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

# 4. Chain and Invoke
chain = prompt_template | model | parser

review_text = ''' I m a long-time patient of this hospital; 
every time I come in, I have to wait over an hour to get seen. 
The staff is nice, and the new renovations look good, 
but I didn't leave work early to wait an hour to be seen
Review by Vaibhav Bhosale '''

result = chain.invoke({"review": review_text})

# 5. Print Results (Access as object attributes)
print(f"Summary: {result.summary}")
print(f"Sentiment: {result.sentiment}")
print(f"Pros: {result.pros}")
