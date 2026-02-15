#langgraph_database.py

from pydantic.v1.fields import FieldInfo as FieldInfoV1
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated, Optional, Any, cast
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_ollama import ChatOllama
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
import sqlite3
import json
import secrets

load_dotenv()

llm = ChatOllama(model='qwen3:8b')

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chat_node(state: ChatState):
    messages = state['messages']
    response = llm.invoke(messages)
    return {"messages": [response]}

conn = sqlite3.connect(database='chatbot.db', check_same_thread=False)
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

# ==================== Helper Functions ====================

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

def get_thread_by_share_token(share_token: str) -> Optional[str]:
    """Get thread_id from share token"""
    cursor = conn.cursor()
    cursor.execute("SELECT thread_id FROM thread_metadata WHERE share_token = ?", (share_token,))
    row = cursor.fetchone()
    return row[0] if row else None

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

# ==================== Graph Setup ====================
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)

graph.add_edge(START, 'chat_node')
graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)