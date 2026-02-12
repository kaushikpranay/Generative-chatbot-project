from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated, Literal
from langchain_core.messages import BaseMessage
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
import os
load_dotenv()
model = os.getenv("OLLAMA_MODEL")
print(model)