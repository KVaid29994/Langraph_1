import json
from typing import List, Dict , Any
from schema import AnswerQuestion, ReviseAnswer
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage , BaseMessage
from langchain_community.tools import TavilySearchResults


# create Tavily search tool

tavily_tool = TavilySearchResults(max_results = 5) 

def excute_tool( state : List[BaseMessage]) -> List[BaseMessage]:
    last_ai_message : AIMessage = state[-1]

    if not hasattr(last_ai_message, "tool_calls") or not last_ai_message.tool_calls:
        return []
    
    tool_messages = []

    for tool_call in last_ai_message.tool_calls:
        if tool_call["name"] in ["AnswerQuestion" ,"ReviseAnswer"]:
            call_id = tool_call["id"]
            search_queries = tool_call["args"].get("search_queries", [])

            query_results = {}
            for query in search_queries:
                result = tavily_tool.invoke(query)
                query_results[query] = result
            
            tool_messages.append(
                ToolMessage(
                    content = json.dumps(query_results), tool_call_id = call_id))
    return tool_messages

'''
mock_state: List[BaseMessage] = [
    HumanMessage(content="Tell me about LangChain and its usage."),
    AIMessage(
        content="LangChain is a framework for developing applications powered by language models. Let me search a bit more to enhance the response.",
        additional_kwargs={
            "tool_calls": [
                {
                    "name": "AnswerQuestion",
                    "args": {
                        "answers": "",
                        "search_queries": [
                            "LangChain use cases",
                            "LangChain architecture",
                            "LangChain vs LlamaIndex"
                        ],
                        "reflection": {"missing": "", "superfluous": ""}
                    },
                    "id": "ajsndsdnjnn"
                }
            ]
        }
    )
]
'''