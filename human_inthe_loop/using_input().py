from typing import TypedDict, Annotated
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, StateGraph, add_messages
from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()

class State(TypedDict):
    messages : Annotated[list, add_messages]

llm = ChatGroq(model = "llama-3.1-8b-instant", temperature= 0.2)

GENERATE_POST = "generate_post"
GET_REVIEW_DECISION = "get_review_decision"
POST = "post"
COLLECT_FEEDBACK = "collect feedback"

def generate_post(state: State):
    return{
        "messages" : [llm.invoke(state['messages'])]
    }

def get_review_decision(state:State):
    post_content = state["messages"][-1].content
    print ("Current linkedin post : \n")

    print (post_content)
    print ("\n")

    decision = input("Post to linkedin (yes/no): ")
    if decision.lower() =="yes":
        return POST
    else:
        return COLLECT_FEEDBACK
    

def post(state: State):
    final_post = state["messages"][-1].content
    print ("Final linkedin post")
    print (final_post)

def collect_feedback(state:State):
    feefback = input("how can I improve this post? ")
    return{
        "messages" : [HumanMessage(content= feefback)]

    }
graph = StateGraph(State)

graph.add_node(GENERATE_POST,generate_post)
graph.add_node(GET_REVIEW_DECISION,get_review_decision)
graph.add_node(COLLECT_FEEDBACK,collect_feedback)
graph.add_node(POST,post)

graph.set_entry_point(GENERATE_POST)
graph.add_edge(POST, END)
graph.add_conditional_edges(GENERATE_POST, get_review_decision)
graph.add_edge(COLLECT_FEEDBACK, GENERATE_POST)

app = graph.compile()

response = app.invoke({"messages" : [HumanMessage(content= "write a linkedin post on how AI agents are taking over content creattion")]})

print (response)