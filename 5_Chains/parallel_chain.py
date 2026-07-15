from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
import os
load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.2-3B-Instruct",
    task="text-generation",
    max_new_tokens=1024, # Increased for structured formatting
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
)


model1 = ChatHuggingFace(llm=llm)
                        

model2 = ChatHuggingFace(llm=llm) 


prompt1= PromptTemplate(
    template='Generate short and simple notes from the following text \n {text}',
    input_variables=['text']
)

prompt2= PromptTemplate(
    template='Generate 5 short questions answers from the following text \n {text}',
    input_variables=['text']
)

prompt3= PromptTemplate(
    template='Marge the provided notes and quiz into a single document \n notes --> {notes} \n  quiz  {quiz}',
    input_variables=['notes', 'quiz']
)

parser = StrOutputParser()


parallel_chain = RunnableParallel({
    'notes':prompt1 | model1 | parser ,
    'quiz': prompt2 | model2 | parser
})


merge_chain = prompt3 | model1 | parser

chain = parallel_chain | merge_chain

text = """
Agentic AI is an artificial intelligence system that can accomplish a specific goal with limited supervision. It consists of AI agents—machine learning models that mimic human decision-making to solve problems in real time. In a multiagent system, each agent performs a specific subtask required to reach the goal and their efforts are coordinated through AI orchestration.

Unlike traditional AI models, which operate within predefined constraints and require human intervention, agentic AI exhibits autonomy, goal-driven behavior and adaptability. The term “agentic” refers to these models’ agency, or, their capacity to act independently and purposefully.

Agentic AI builds on generative AI (gen AI) techniques by using large language models (LLMs) to function in dynamic environments. While generative models focus on creating content based on learned patterns, agentic AI extends this capability by applying generative outputs toward specific goals. A generative AI model like OpenAI’s ChatGPT might produce text, images or code, but an agentic AI system can use that generated content to complete complex tasks autonomously by calling external tools. Agents can, for example, not only tell you the best time to climb Mt. Everest given your work schedule, it can also book you a flight and a hotel.

Agentic systems have many advantages over their generative predecessors, which are limited by the information contained in the datasets upon which models are trained.

Autonomous
The most important advancement of agentic systems is that they allow for autonomy to perform tasks without constant human oversight. Agentic systems can maintain long-term goals, manage multistep problem-solving tasks and track progress over time.

Proactive
Agentic systems provide the flexibility of LLMs, which can generate responses or actions based on nuanced, context-dependent understanding, with the structured, deterministic and reliable features of traditional programming. This approach allows agents to “think” and “do” in a more human-like fashion.

LLMs by themselves can’t directly interact with external tools or databases or set up systems to monitor and collect data in real time, but agents can. Agents can search the web, call application programming interfaces (APIs) and query databases, then use this information to make decisions and take actions.

Specialized
Agents can specialize in specific tasks. Some agents are simple, performing a single repetitive task reliably. Others can use perception and draw on memory to solve more complex problems. An agentic architecture might consist of a “conductor” model powered by an LLM that oversees tasks and decisions and supervises other, simpler agents. Such architectures are ideal for sequential workflows but are vulnerable to bottlenecks. Other architectures are more horizontal, with agents working in harmony as equals in a decentralized fashion, but this architecture can be slower than a vertical hierarchy. Different AI applications demand different architectures.

Adaptable
Agents can learn from their experiences, take in feedback and adjust their behavior. With the right guardrails, agentic systems can improve continuously. Multiagent systems possess the scalability to eventually handle broadly scoped initiatives.

Intuitive
Because agentic systems are powered by LLMs, users can engage with them with natural language prompts. This means that entire software interfaces—think of the many tabs, dropdowns, charts, sliders, pop-ups and other UI elements involved in the SaaS platform of one’s choice—can be replaced by simple language or voice commands. Theoretically, any software user experience can now be reduced to “talking” with an agent, who can fetch the information one needs and take action based on that information. This productivity benefit can barely be overstated, when one considers the time it takes for workers to learn and master new interfaces and tools.
"""
 
result = chain.invoke({'text':text})

print(result)

chain.get_graph().print_ascii()