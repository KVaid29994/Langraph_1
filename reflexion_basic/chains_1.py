# chains_1.py
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from schema import AnswerQuestion
from langchain_core.output_parsers.openai_tools import PydanticToolsParser
from langchain_core.runnables import RunnableLambda
import datetime
import json
import re
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

load_dotenv()

pydantic_parser = PydanticToolsParser(tools=[AnswerQuestion])
# Setup HuggingFace Model
llm = ChatOpenAI()

# Actor Agent Prompt
actor_prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an expert AI researcher.
Current time: {time}

1. {first_instruction}
2. Reflect and critique your answer. Be severe to maximize improvement.
3. After the reflection, **list 1-3 search queries separately** for researching improvements. Do not include them inside the reflection.

"""
        ),
        MessagesPlaceholder(variable_name="messages"),
        (
            "system",
            "Answer the user's question above using the required JSON format."
        ),
    ]
).partial(time=lambda: datetime.datetime.now().isoformat())

first_responder_prompt_template = actor_prompt_template.partial(
    first_instruction="Provide a detailed ~250 word answer"
)

first_responder_chain = first_responder_prompt_template | llm.bind_tools(tools=[AnswerQuestion], tool_choice= "AnswerQuestion") | pydantic_parser

response= first_responder_chain.invoke({"messages": [HumanMessage(content = "Write me a blog post on how AI can help small businness to grow")]})

print (response)

