# langsmith.py

from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
import os

from langsmith import traceable
from langsmith.run_helpers import get_current_run_tree

# ========================
# ENV
# ========================
load_dotenv()

MODEL_NAME = os.getenv("OLLAMA_MODEL", "qwen3:8b")

llm = ChatOllama(
    model=MODEL_NAME
)

# ========================
# STATE
# ========================
class ChatState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]
    thread_id: str  # optional metadata if you pass it from config


# ========================
# NODE
# ========================
@traceable(
    run_type="llm",
    name="chatbot_response",
    tags=["production", "chatbot"]
)
def chat_node(state: ChatState):

    # invoke LLM with full message history
    messages = state.get("messages", [])
    response = llm.invoke(messages)

    # attach metadata to LangSmith run
    run_tree = get_current_run_tree()
    if run_tree:
        run_tree.extra = {
            "message_count": len(state.get("messages", [])),
            "thread_id": state.get("thread_id"),
            "model": MODEL_NAME,
        }

    # return new assistant message
    return {"messages": [response]}


# ========================
# CHECKPOINTER
# ========================
checkpointer = InMemorySaver()

# ========================
# GRAPH BUILD
# ========================
graph = StateGraph(ChatState)

graph.add_node("chat_node", chat_node)

graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)
