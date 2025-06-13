from typing import List, Sequence
from dotenv import load_dotenv

from langgraph.graph import  END, MessageGraph
from langchain_core.messages import BaseMessage, HumanMessage
from chains import generation_chain, reflection_chain

load_dotenv()

graph = MessageGraph()

REFLECT = "reflect"
GENERATE = "generate"

def generate_node(state):
    return generation_chain.invoke({"messages": state})

def reflect_node(state):
    response =  reflection_chain.invoke({"messages": state})
    return [HumanMessage(content=response)]


graph.add_node(GENERATE, generate_node)
graph.add_node(REFLECT, reflect_node)

graph.set_entry_point(GENERATE)

def should_continue(state):
    if (len(state) >4):
        return END
    else:
        return REFLECT
    

graph.add_conditional_edges(GENERATE,should_continue)

graph.add_edge(REFLECT, GENERATE)

app = graph.compile()

# print (app.get_graph().draw_mermaid())
# app.get_graph().print_ascii()

response = app.invoke(HumanMessage(content = "Ai agents taking over content creation"))
print (response)

# Extract only non-empty human messages
relevant_messages = [msg.content for msg in response if isinstance(msg, HumanMessage) and msg.content.strip() != ""]

# Get the last one
final_tweet = relevant_messages[-1]

print("📝 Final Tweet:")
print(final_tweet)


'''
🧪 Output Sample (Simplified Summary)
Initial Prompt: "AI agents taking over content creation"

First Tweet: A hopeful post about collaboration with AI

Reflection: Too optimistic, revise with more critique

Next Tweet: More skeptical tone, adds hashtags

Reflection: Still too soft, adds personal story of journalist losing job

Tweet: Final concise version with emojis and hashtags
'''

