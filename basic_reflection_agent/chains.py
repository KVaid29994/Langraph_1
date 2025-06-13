from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="conversational"
)

model = ChatHuggingFace(llm=llm)

generation_prompt = ChatPromptTemplate.from_messages([
    ("system", 
     "You are a twitter techie assistant tasked with writing excellent twitter posts. "
     "Generate the best twitter post possible for the user's request. "
     "If the user provides critique, respond with a revised version of your previous attempt."),
    MessagesPlaceholder(variable_name="messages")
])


reflection_prompt = ChatPromptTemplate.from_messages([
    ("system", 
     "You are a viral Twitter influencer grading a tweet. "
     "Generate critique and recommendations for the user's tweet. "
     "Always provide detailed feedback, including suggestions on length, virality, tone, and formatting."),
    MessagesPlaceholder(variable_name="messages")
])

generation_chain = generation_prompt | llm
reflection_chain = reflection_prompt | llm


