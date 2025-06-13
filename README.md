# Langraph_1

# 🤖 LangGraph Agents Playground

Welcome to the **LangGraph Agents Playground**, a hands-on project where I explore autonomous agent patterns using the [LangGraph](https://python.langgraph.org/) library. This repo showcases different types of AI agents with LangChain + HuggingFace integrations.

---

🚀 What is LangGraph?
LangGraph is a stateful, graph-based orchestration framework designed for building autonomous agents and multi-step reasoning flows using LangChain.

 🔁 Reflection Agent
A two-node cycle that allows the model to self-critique and revise output.

♻️ Reflexion Agent
Implements the Reflexion pattern, where the agent reflects on past failures to guide future generations.

Maintains memory of failures and success

Learns from mistakes in multiple iterations

Suitable for coding, reasoning, or tweet composition tasks

🧠 Tech Stack
LangGraph

LangChain

HuggingFace Inference Endpoints

dotenv for config management

ChatPromptTemplate and MessagesPlaceholder for dynamic prompting