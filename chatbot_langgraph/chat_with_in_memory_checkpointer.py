from typing import TypedDict, Annotated
from langgraph.graph import END, StateGraph,add_messages
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, HumanMessage
from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()
memory = MemorySaver()

llm = ChatGroq(model = "llama-3.1-8b-instant", temperature= 0.2)

class BasicChatState(TypedDict):
    messages : Annotated[list, add_messages]

def chatbot(state: BasicChatState):
    return{"messages":[llm.invoke(state["messages"])]}

graph = StateGraph(BasicChatState)

graph.add_node("chatbot", chatbot)
graph.set_entry_point("chatbot")
graph.add_edge("chatbot", END)

app = graph.compile(checkpointer=memory)

config = {"configurable":{"thread_id":1}}

response1 = app.invoke({"messages": HumanMessage(content="Hi I am Kashish")},config=config)
response2 = app.invoke({"messages": HumanMessage(content="what is my name?")},config=config)

print (app.get_state(config=config))
# print (response1)
# print ("----"*10)
# print (response2)