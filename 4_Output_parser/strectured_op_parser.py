from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
# 2026lanhchain removed function and updated
#from langchain_core.output_parsers import StructuredOutputParser, ResponseSchema
#from langchain.output_parsers import StructuredOutputParser , ResponseSchema
# New versin
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate

import os
load_dotenv() 

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.2-3B-Instruct",
    task="text-generation",
    max_new_tokens=1024, # Increased for structured formatting
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
)


model = ChatHuggingFace(llm=llm)


class Facts(BaseModel):
    fact_1: str = Field(description="Fact 1 about topic")
    fact_2: str = Field(description="Fact 2 about topic")
    fact_3: str = Field(description="Fact 3 about topic")


parser = PydanticOutputParser(pydantic_object=Facts)

template = PromptTemplate(
    template=(
        "Give exactly 3 facts about {topic}.\n\n"
        "Return ONLY a valid JSON object with the following keys:\n"
        "fact_1, fact_2, fact_3\n\n"
        "Do NOT include schema, properties, descriptions, or explanations.\n"
        "Only return the filled JSON.\n\n"
        "{format_instructions}"
    ),
    input_variables=["topic"],
    partial_variables={
        "format_instructions": parser.get_format_instructions()
    }
)


prompt = template.invoke({"topic": "Law of attraction"})
result = model.invoke(prompt)

print("RAW MODEL OUTPUT:\n", result.content)
print('_' * 80)

final_result = parser.parse(result.content)
print("\nPARSED OUTPUT:\n", final_result)
