from typing import TypedDict, Annotated
from langgraph.graph import END, StateGraph,add_messages
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, HumanMessage
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3

load_dotenv()

sqlite_conn = sqlite3.connect("checkpoint.sqlite", check_same_thread=False)
memory = SqliteSaver(sqlite_conn)

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


while True:
    user_input = input("User: ")
    if (user_input in ["end","exit"]):
        break
    else:
        result =  app.invoke({"messages": HumanMessage(content=user_input)},config=config)

        print ("AI: " + result["messages"][-1].content)