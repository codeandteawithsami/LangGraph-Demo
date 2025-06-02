import os 
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from typing import TypedDict, List, Union 
from langgraph.graph import StateGraph, START,END 
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

class AgentState(TypedDict):
    messages : List[Union [HumanMessage,AIMessage]]

llm = ChatOpenAI(model="gpt-4o-mini")

def process(state : AgentState) -> AgentState:
    """This node will solve the request you input"""
    response = llm.invoke(state["messages"])
    
    state["messages"].append(AIMessage(content=response.content))
    print(f"AI: {response.content}")
    # print("CURRENT STATE: ",state['messages'])
    
    return state 

graph = StateGraph(AgentState)
graph.add_node("process",process)
graph.add_edge(START,"process")
graph.add_edge("process",END)
agent = graph.compile()

conversation_history = []

user_input = input("Sami : ")
while user_input != "exit":
    conversation_history.append(HumanMessage(content=user_input))
    result = agent.invoke({"messages":conversation_history})
    
    # print(result["messages"])
    conversation_history = result["messages"]
    user_input = input("Sami : ")
    

with open("logging.txt", "w") as file:
    file.write("Your Conersation Log:\n\n\n")
    for message in conversation_history:
        if isinstance(message, HumanMessage):
            file.write(f"Samiullah Saleem: {message.content}\n")
        elif isinstance(message, AIMessage):
            file.write(f"AI: {message.content}\n\n")
    file.write("End of conversation")
    
print("Conversation histroy saved to logging.txt")