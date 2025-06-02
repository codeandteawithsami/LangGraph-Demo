from dotenv import load_dotenv
from langchain_core.tools import tool 
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode 
from langgraph.graph import StateGraph, END 
from langchain_core.messages import ToolMessage # Passes data back to LLM after it calls a tools such as the content 
from langchain_core.messages import BaseMessage # The Foundational class for all message types in LangGraph
from langgraph.graph.message import add_messages 
from typing import Annotated, Sequence, TypedDict 
from langchain_core.messages import SystemMessage # Message for providing instrutions to the LLM


load_dotenv()

# Annotated - provides additional context without affecting the type itself.
# Sequence - To automatically handle the state updates for sequences such as by adding new messages to a chat history.

# Reducer Function
# Rule that controls how updates from nodes are combined with the existing state.
# Tells us how to merge new data into the current state

# without a reducer, updates would have replaced the existing value entirely! 

class AgentState(TypedDict):
    messages : Annotated[Sequence[BaseMessage], add_messages]
    
    
@tool 
def add(a: int, b:int):
    """This is an addition function that adds 2 numbers together"""
    
    return a + b

@tool 
def substract(a: int, b:int):
    """This is an Substraction function that adds 2 numbers together"""
    
    return a - b

@tool 
def multiply(a: int, b:int):
    """This is an Multiplication function that adds 2 numbers together"""
    
    return a * b

tools = [add,substract]

model = ChatOpenAI(model = "gpt-4o-mini").bind_tools(tools)

def model_call(state:AgentState) -> AgentState:
    
    system_prompt = SystemMessage(content=
                "You are my AI assistant, please answer my query to the best of your ability."                                  
                                  )
    
    response = model.invoke([system_prompt] + state["messages"])
    return {'messages':[response]}

def should_continue(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]
    if not last_message.tool_calls:
        return "end"
    else:
        return "continue"

graph = StateGraph(AgentState)
graph.add_node("our_agent", model_call)

tool_node = ToolNode(tools=tools)
graph.add_node("tools", tool_node)

graph.set_entry_point("our_agent")

graph.add_conditional_edges(
    "our_agent",
    should_continue,
    {
        "continue":"tools",
        "end": END
    }
)

graph.add_edge("tools", "our_agent")

app = graph.compile()

def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()
       
    
# inputs = {"messages": [("user", "Add 40 + 12 .Mul 3 * 4. Add 12 + 12 . Add 89 - 21")]} 
inputs = {"messages": [("user", "Add 40 + 12 and then multiply the result by 6. Also tell me a joke please.")]} 
print_stream(app.stream(inputs, stream_mode="values"))

    
            