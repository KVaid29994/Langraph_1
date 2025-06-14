from langchain.prompts import ChatPromptTemplate , MessagesPlaceholder
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from schema import AnswerQuestion, ReviseAnswer
from langchain_core.output_parsers import PydanticToolsParser , JsonOutputKeyToolsParser
load_dotenv()
import datetime
from langchain_core.messages import HumanMessage

pydantic_parser = PydanticToolsParser(tools=[AnswerQuestion])

parser = JsonOutputKeyToolsParser(key_name='AnswerQuestion')

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)
# Actor Agent Prompt 
actor_prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are expert AI researcher.
Current time: {time}

1. {first_instruction}
2. Reflect and critique your answer. Be severe to maximize improvement.
3. After the reflection, **list 1-3 search queries separately** for researching improvements. Do not include them inside the reflection.
""",
        ),
        MessagesPlaceholder(variable_name="messages"),
        ("system", "Answer the user's question above using the required format."),
    ]
).partial(time=lambda: datetime.datetime.now().isoformat(),
)

first_responder_prompt_template = actor_prompt_template.partial(
    first_instruction="Provide a detailed ~250 word answer"
)


first_responder_chain = first_responder_prompt_template | llm | parser

response = first_responder_chain.invoke({'messages':[HumanMessage(content= "write me a blog post on how small business can leverage AI to grow ")]})

print (response)
