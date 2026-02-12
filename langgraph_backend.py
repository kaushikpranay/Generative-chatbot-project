from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated, Literal
from langchain_core.messages import BaseMessage
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
import os
from langsmith import traceable
from langsmith.run_helpers import get_current_run_tree
load_dotenv()

llm = ChatOllama(model = os.getenv("OLLAMA_MODEL", "deepseek-r1:1.5b"))
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

@traceable(
    run_type="llm",
    name="chatbot_response",
    tags=["production", "chatbot"]
)

def chat_node(state: ChatState):

    messages = state['messages']
    response = llm.invoke(state["messages"])

    run_tree = get_current_run_tree()
    if run_tree:
        run_tree.extra = {
            "message_count": len(state["message"]),
            "thread_id": state.get("thread_id"),
            "model": "gpt-oss:20b"
        }
    return {"messages": [response]}


#checkPointer
checkpointer = InMemorySaver()

#graphs
graph = StateGraph(ChatState)

graph.add_node("chat_node", chat_node)

graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)
chatbot = graph.compile(checkpointer=checkpointer)

