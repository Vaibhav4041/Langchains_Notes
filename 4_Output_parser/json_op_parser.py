from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import os
load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.2-3B-Instruct",
    task="text-generation",
    max_new_tokens=1024, # Increased for structured formatting
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
)


model = ChatHuggingFace(llm=llm)

parser = JsonOutputParser()

template = PromptTemplate(
    template="Give me the name, age and city of a fictional person. \n {format_instruction}",
    input_variables=[],
    partial_variables={
        "format_instruction": parser.get_format_instructions()}
)

prompt = template.format()


result = model.invoke(prompt)


final_result = parser.parse(result.content)
print(final_result)

