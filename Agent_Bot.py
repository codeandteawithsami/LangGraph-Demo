import os
from dotenv import load_dotenv
from typing import Dict,List,TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START,END 

load_dotenv()

class AgentState(TypedDict):
    messages : List[HumanMessage]
    
    
llm = ChatOpenAI(model="gpt-4o-mini")
def process(state: AgentState)-> AgentState:
    response = llm.invoke(state["messages"])
    print(f"\n AI :{response.content}")
    return state


graph = StateGraph(AgentState)
graph.add_node("process",process)
graph.add_edge(START,"process")
graph.add_edge("process",END)
agent = graph.compile()

user_input = input("Enter : ")
while user_input != "exit":
    agent.invoke({"messages":[HumanMessage(content=user_input)]})
    user_input = input("Enter : ")
    