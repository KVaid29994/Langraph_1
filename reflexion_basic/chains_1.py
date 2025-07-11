# chains_1.py
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from schema import AnswerQuestion , ReviseAnswer
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

first_responder_chain = first_responder_prompt_template | llm.bind_tools(tools=[AnswerQuestion], tool_choice= "AnswerQuestion")

validator = PydanticToolsParser(tools=[AnswerQuestion])

revise_instructios = '''
revise your previous answer using the new information.
    - you must use your previous crtique to add important information to your answer
    - you must include numerical citations in your revised answer to ensure it can be verified
    - add refernece to the bottom of your answer to ensure it can be verified, In form of
        - 1) https://example.com
        - 1) https://examplew.com

    - you should previous critique to ensure removal of superflous information from your answer
    Make sure your answer is not more than 250 words

'''
revisor_chain = actor_prompt_template.partial(first_instruction = revise_instructios) | llm.bind_tools(tools =[ReviseAnswer], tool_choice="ReviseAnswer")



response= first_responder_chain.invoke({"messages": [HumanMessage(content = "Write me a blog post on how AI can help small businness to grow")]})

print (response)

