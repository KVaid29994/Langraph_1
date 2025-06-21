from langchain_openai import ChatOpenAI
from langchain.agents import tool, create_react_agent
import datetime
from langchain import hub
from langchain_community.tools import TavilySearchResults
from dotenv import load_dotenv
load_dotenv()

llm = ChatOpenAI(model="gpt-4o")

search_tool = TavilySearchResults(search_depth = "basic")

@tool
def get_system_time(format : str = "%Y-%m-%d %H:%M:%S"):
    "returns current date and time in the specified format"
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime(format)
    return formatted_time

tools = [search_tool, get_system_time]

react_prompt = hub.pull("hwchase17/react")

react_agent_runnable = create_react_agent(tools=tools, llm = llm, prompt=react_prompt)
    