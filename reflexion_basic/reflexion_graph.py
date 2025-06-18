from typing import List
from langchain_core.messages import BaseMessage, ToolMessage
from langgraph.graph import END, MessageGraph
from chains_1 import revisor_chain, first_responder_chain
from execute_tools import excute_tool

graph = MessageGraph()
Max_iterations = 2

graph.add_node("draft", first_responder_chain)
graph.add_node("execute_tools", excute_tool)
graph.add_node("reviser", revisor_chain)

graph.add_edge("draft", "execute_tools")
graph.add_edge("execute_tools", "reviser")

def event_loop(state : List[BaseMessage]) -> str:
    count_tools_visits = sum(isinstance(item, ToolMessage) for item in state)
    num_iterations = count_tools_visits
    if num_iterations > Max_iterations:
        return END
    return "execute_tools"

graph.add_conditional_edges("reviser", event_loop)

graph.set_entry_point("draft")

app = graph.compile()

print (app.get_graph().draw_mermaid())

response = app.invoke("Write about how small business can leverage AI to grow?")

print (response)
print (response[-1].tool_calls[0]["args"]["answer"])