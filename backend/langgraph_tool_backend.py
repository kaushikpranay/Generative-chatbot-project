#langgraph_tool_backend.py

from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_ollama import ChatOllama
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import DuckDuckGoSearchRun
from typing import Any, cast
from langchain_core.tools import tool
from dotenv import load_dotenv
import sqlite3
import requests
import secrets

load_dotenv()

#-----------
#1. llm
#-------------
llm = ChatOllama(model='qwen3:8b')

#2. tools

search_tool = DuckDuckGoSearchRun()

@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """
    Perform a basic arithmetic operation on two numbers.
    Supported operations: add, sub, mul, div
    """
    try:
        if operation == "add":
            result = first_num + second_num
        elif operation == "sub":
            result = first_num - second_num
        elif operation == "mul":
            result = first_num * second_num
        elif operation == "div":
            if second_num == 0:
                return {"error": "Division by zero is not allowed"}
            result = first_num / second_num
        else:
            return {"error": f"Unsupported operation '{operation}'"}
        
        return {"first_num": first_num, "second_num": second_num, "operation": operation, "result": result}
    except Exception as e:
        return {"error": str(e)}

@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA') 
    using Alpha Vantage with API key in the URL.
    """
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey=C9PE94QUEW9VWGFM"
    r = requests.get(url)
    return r.json()

tools = [search_tool, get_stock_price, calculator]
llm_with_tools = llm.bind_tools(tools)

# 3. state

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# 4. Nodes

def chat_node(state: ChatState):
    """
    LLM node that may answer or request a tool call.
    """
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

tool_node = ToolNode(tools)

# 5. Checkpointer

conn = sqlite3.connect(database="chatbot.db", check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)

# ==================== Thread Metadata Table ====================
cursor = conn.cursor()
cursor.execute("""
    CREATE TABLE IF NOT EXISTS thread_metadata (
        thread_id TEXT PRIMARY KEY,
        display_name TEXT,
        is_deleted INTEGER DEFAULT 0,
        share_token TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
""")
conn.commit()

#6. Graph

graph = StateGraph(ChatState)

graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge('tools', 'chat_node')

chatbot = graph.compile(checkpointer=checkpointer)

# 7. Helper Functions

def retrieve_all_threads():
    """Get all non-deleted threads"""
    cursor = conn.cursor()
    cursor.execute("""
        SELECT thread_id FROM thread_metadata 
        WHERE is_deleted = 0
        ORDER BY created_at DESC
    """)
    rows = cursor.fetchall()
    return [row[0] for row in rows]

def get_thread_display_name(thread_id: str) -> str:
    """Get display name for a thread, returns thread_id if not set"""
    cursor = conn.cursor()
    cursor.execute("SELECT display_name FROM thread_metadata WHERE thread_id = ?", (str(thread_id),))
    row = cursor.fetchone()
    return row[0] if row and row[0] else str(thread_id)

def set_thread_display_name(thread_id: str, display_name: str):
    """Set or update display name for a thread"""
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO thread_metadata (thread_id, display_name)
        VALUES (?, ?)
        ON CONFLICT(thread_id) DO UPDATE SET display_name = ?
    """, (str(thread_id), display_name, display_name))
    conn.commit()

def auto_name_thread_from_first_message(thread_id: str, first_message: str):
    """Auto-name thread from first 20 chars of first message"""
    if len(first_message) <= 20:
        display_name = first_message
    else:
        display_name = first_message[:20] + "..."
    set_thread_display_name(thread_id, display_name)

def delete_thread(thread_id: str):
    """Mark thread as deleted (soft delete)"""
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO thread_metadata (thread_id, is_deleted)
        VALUES (?, 1)
        ON CONFLICT(thread_id) DO UPDATE SET is_deleted = 1
    """, (str(thread_id),))
    conn.commit()

def rename_thread(thread_id: str, new_name: str):
    """Rename a thread"""
    set_thread_display_name(thread_id, new_name)

def generate_share_token(thread_id: str) -> str:
    """Generate a unique share token for a thread"""
    cursor = conn.cursor()
    cursor.execute("SELECT share_token FROM thread_metadata WHERE thread_id = ?", (str(thread_id),))
    row = cursor.fetchone()
    
    if row and row[0]:
        return row[0]
    
    # Generate new token
    token = secrets.token_urlsafe(16)
    cursor.execute("""
        INSERT INTO thread_metadata (thread_id, share_token)
        VALUES (?, ?)
        ON CONFLICT(thread_id) DO UPDATE SET share_token = ?
    """, (str(thread_id), token, token))
    conn.commit()
    return token

def export_thread_conversation(thread_id: str) -> dict:
    """Export thread conversation as JSON"""
    state = chatbot.get_state(config=cast(Any, {'configurable': {'thread_id': str(thread_id)}}))
    messages = state.values.get('messages', [])
    
    conversation = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            role = 'user'
        else:
            role = 'assistant'
        conversation.append({'role': role, 'content': msg.content})
    
    return {
        'thread_id': str(thread_id),
        'display_name': get_thread_display_name(thread_id),
        'conversation': conversation
    }

def ensure_thread_exists(thread_id: str):
    """Ensure thread exists in metadata table"""
    cursor = conn.cursor()
    cursor.execute("""
        INSERT OR IGNORE INTO thread_metadata (thread_id, is_deleted)
        VALUES (?, 0)
    """, (str(thread_id),))
    conn.commit()