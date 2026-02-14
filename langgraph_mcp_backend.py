from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_ollama import ChatOllama
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import DuckDuckGoSearchRun
from typing import Any, cast
from langchain_core.tools import tool, BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from dotenv import load_dotenv
import aiosqlite
import requests
import asyncio
import threading
import secrets

load_dotenv()

_ASYNC_LOOP = asyncio.new_event_loop()
_ASYNC_THREAD = threading.Thread(target=_ASYNC_LOOP.run_forever, daemon=True)
_ASYNC_THREAD.start()

def _submit_async(coro):
    return asyncio.run_coroutine_threadsafe(coro, _ASYNC_LOOP)

def run_async(coro):
    return _submit_async(coro).result()

def submit_async_task(coro):
    """Schedule a coroutine on the backend event loop."""
    return _submit_async(coro)

#1. LLM
llm = ChatOllama(model="qwen3:8b")

#2.tools
search_tool = DuckDuckGoSearchRun()

@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA') 
    using Alpha Vantage with API key in the URL.
    """
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey=C9PE94QUEW9VWGFM"
    r = requests.get(url)
    return r.json()

client = MultiServerMCPClient(
    {
        "arith": {
            "transport": "stdio",
            "command": "python3",
            "args": ["/Users/prana/Desktop/mcp-math-server/main.py"],
        },
        "expense": {
            "transport": "streamable_http",
            "url": "https://splendid-gold-dingo.fastmcp.app/mcp"
        }
    }
)

def load_mcp_tools() -> list[BaseTool]:
    try:
        return run_async(client.get_tools())
    except Exception:
        return []

mcp_tools = load_mcp_tools()

tools = [search_tool, get_stock_price, *mcp_tools]
llm_with_tools = llm.bind_tools(tools) if tools else llm

#3.state
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# -------------------
# 4. Nodes
# -------------------
async def chat_node(state: ChatState):
    """LLM node that may answer or request a tool call."""
    messages = state["messages"]
    response = await llm_with_tools.ainvoke(messages)
    return {"messages": [response]}

tool_node = ToolNode(tools) if tools else None

# -------------------
# 5. Checkpointer
# -------------------

async def _init_checkpointer():
    conn = await aiosqlite.connect(database="chatbot.db")
    # Create metadata table
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS thread_metadata (
            thread_id TEXT PRIMARY KEY,
            display_name TEXT,
            is_deleted INTEGER DEFAULT 0,
            share_token TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    await conn.commit()
    return AsyncSqliteSaver(conn)

checkpointer = run_async(_init_checkpointer())

# -------------------
# 6. Graph
# -------------------
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")

if tool_node:
    graph.add_node("tools", tool_node)
    graph.add_conditional_edges("chat_node", tools_condition)
    graph.add_edge("tools", "chat_node")
else:
    graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)

# -------------------
# 7. Helper Functions (Async)
# -------------------

async def _alist_threads():
    all_threads = []
    conn = checkpointer.conn
    async with conn.execute("""
        SELECT thread_id FROM thread_metadata 
        WHERE is_deleted = 0
        ORDER BY created_at DESC
    """) as cursor:
        async for row in cursor:
            all_threads.append(row[0])
    return all_threads

def retrieve_all_threads():
    return run_async(_alist_threads())

async def _get_thread_display_name(thread_id: str) -> str:
    conn = checkpointer.conn
    async with conn.execute(
        "SELECT display_name FROM thread_metadata WHERE thread_id = ?", 
        (str(thread_id),)
    ) as cursor:
        row = await cursor.fetchone()
        return row[0] if row and row[0] else str(thread_id)

def get_thread_display_name(thread_id: str) -> str:
    return run_async(_get_thread_display_name(thread_id))

async def _set_thread_display_name(thread_id: str, display_name: str):
    conn = checkpointer.conn
    await conn.execute("""
        INSERT INTO thread_metadata (thread_id, display_name)
        VALUES (?, ?)
        ON CONFLICT(thread_id) DO UPDATE SET display_name = ?
    """, (str(thread_id), display_name, display_name))
    await conn.commit()

def set_thread_display_name(thread_id: str, display_name: str):
    run_async(_set_thread_display_name(thread_id, display_name))

def auto_name_thread_from_first_message(thread_id: str, first_message: str):
    if len(first_message) <= 20:
        display_name = first_message
    else:
        display_name = first_message[:20] + "..."
    set_thread_display_name(thread_id, display_name)

async def _delete_thread(thread_id: str):
    conn = checkpointer.conn
    await conn.execute("""
        INSERT INTO thread_metadata (thread_id, is_deleted)
        VALUES (?, 1)
        ON CONFLICT(thread_id) DO UPDATE SET is_deleted = 1
    """, (str(thread_id),))
    await conn.commit()

def delete_thread(thread_id: str):
    run_async(_delete_thread(thread_id))

def rename_thread(thread_id: str, new_name: str):
    set_thread_display_name(thread_id, new_name)

async def _generate_share_token(thread_id: str) -> str:
    conn = checkpointer.conn
    async with conn.execute(
        "SELECT share_token FROM thread_metadata WHERE thread_id = ?", 
        (str(thread_id),)
    ) as cursor:
        row = await cursor.fetchone()
        
        if row and row[0]:
            return row[0]
    
    # Generate new token
    token = secrets.token_urlsafe(16)
    await conn.execute("""
        INSERT INTO thread_metadata (thread_id, share_token)
        VALUES (?, ?)
        ON CONFLICT(thread_id) DO UPDATE SET share_token = ?
    """, (str(thread_id), token, token))
    await conn.commit()
    return token

def generate_share_token(thread_id: str) -> str:
    return run_async(_generate_share_token(thread_id))

async def _export_thread_conversation(thread_id: str) -> dict:
    state = await chatbot.aget_state(config=cast(Any, {'configurable': {'thread_id': str(thread_id)}}))
    messages = state.values.get('messages', [])
    
    conversation = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            role = 'user'
        else:
            role = 'assistant'
        conversation.append({'role': role, 'content': msg.content})
    
    display_name = await _get_thread_display_name(thread_id)
    
    return {
        'thread_id': str(thread_id),
        'display_name': display_name,
        'conversation': conversation
    }

def export_thread_conversation(thread_id: str) -> dict:
    return run_async(_export_thread_conversation(thread_id))

async def _ensure_thread_exists(thread_id: str):
    conn = checkpointer.conn
    await conn.execute("""
        INSERT OR IGNORE INTO thread_metadata (thread_id, is_deleted)
        VALUES (?, 0)
    """, (str(thread_id),))
    await conn.commit()

def ensure_thread_exists(thread_id: str):
    run_async(_ensure_thread_exists(thread_id))