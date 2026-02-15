# langgraph_backend.py

from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
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

MODEL_NAME = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")

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




#=================
#Summary
#================


SUMMARIZATION_THRESHOLD = 10  # Summarize every 10 messages

def should_summarize(messages: list[BaseMessage]) -> bool:
    """Check if we need to summarize"""
    # Count only user+assistant pairs (ignore system messages)
    conversation_messages = [m for m in messages if not isinstance(m, SystemMessage)]
    return len(conversation_messages) >= SUMMARIZATION_THRESHOLD

def summarize_conversation(messages: list[BaseMessage], llm: ChatOllama) -> str:
    """Generate summary of conversation history"""
    # Filter out system messages and get only user/assistant dialogue
    conversation_messages = [m for m in messages if not isinstance(m, SystemMessage)]
    
    # Build prompt for summarization
    conversation_text = "\n".join([
        f"{'User' if i % 2 == 0 else 'Assistant'}: {msg.content}"
        for i, msg in enumerate(conversation_messages[:-2])  # Exclude last 2 messages
    ])
    
    summary_prompt = f"""Summarize this conversation concisely. Focus on:
- Key topics discussed
- Important information shared
- User's main questions/concerns

Conversation:
{conversation_text}

Summary:"""
    
    response = llm.invoke([HumanMessage(content=summary_prompt)])
    return response.content

# Update your ChatState
class ChatState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]
    thread_id: str
    conversation_summary: str  # NEW: Store summary here

# Update chat_node
def chat_node(state: ChatState):
    messages = state.get("messages", [])
    summary = state.get("conversation_summary", "")
    
    # Check if summarization needed
    if should_summarize(messages) and not summary:
        # Generate summary (this happens in background, user never sees it)
        summary = summarize_conversation(messages, llm)
        state["conversation_summary"] = summary
    
    # Build context for LLM
    context_messages = []
    
    # Add summary as system context if it exists
    if summary:
        context_messages.append(
            SystemMessage(content=f"Previous conversation summary:\n{summary}")
        )
    
    # Add last 4 messages (2 exchanges) for immediate context
    recent_messages = messages[-4:] if len(messages) > 4 else messages
    context_messages.extend(recent_messages)
    
    # Get response
    response = llm.invoke(context_messages)
    
    return {"messages": [response]}