from __future__ import annotations

import os
import sqlite3
import tempfile
import secrets
from typing import Annotated, Any, Dict, Optional, TypedDict
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.vectorstores import FAISS
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
import requests

load_dotenv()
llm = ChatOllama(model="qwen3:8b")
embeddings = OllamaEmbeddings(model="qwen3-embedding:8b")

# -------------------
# 2. PDF retriever store (per thread)
# -------------------
_THREAD_RETRIEVERS: Dict[str, Any] = {}
_THREAD_METADATA: Dict[str, dict] = {}

def _get_retriever(thread_id: Optional[str]):
    """Fetch the retriever for a thread if available."""
    if thread_id and thread_id in _THREAD_RETRIEVERS:
        return _THREAD_RETRIEVERS[thread_id]
    return None

def ingest_pdf(file_bytes: bytes, thread_id: str, filename: Optional[str] = None) -> dict:
    """
    Build a FAISS retriever for the uploaded PDF and store it for the thread.
    Returns a summary dict that can be surfaced in the UI.
    """
    if not file_bytes:
        raise ValueError("No bytes received for ingestion.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(file_bytes)
        temp_path = temp_file.name

    try:
        loader = PyPDFLoader(temp_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_documents(docs)

        vector_store = FAISS.from_documents(chunks, embeddings)
        retriever = vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": 4}
        )

        _THREAD_RETRIEVERS[str(thread_id)] = retriever
        _THREAD_METADATA[str(thread_id)] = {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(chunks),
        }

        return {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(chunks),
        }
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass

# -------------------
# 3. Tools
# -------------------
search_tool = DuckDuckGoSearchRun(region="us-en")

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

        return {
            "first_num": first_num,
            "second_num": second_num,
            "operation": operation,
            "result": result,
        }
    except Exception as e:
        return {"error": str(e)}

@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA') 
    using Alpha Vantage with API key in the URL.
    """
    url = (
        "https://www.alphavantage.co/query"
        f"?function=GLOBAL_QUOTE&symbol={symbol}&apikey=C9PE94QUEW9VWGFM"
    )
    r = requests.get(url)
    return r.json()

@tool
def rag_tool(query: str, thread_id: Optional[str] = None) -> dict:
    """
    Retrieve relevant information from the uploaded PDF for this chat thread.
    Always include the thread_id when calling this tool.
    """
    retriever = _get_retriever(thread_id)
    if retriever is None:
        return {
            "error": "No document indexed for this chat. Upload a PDF first.",
            "query": query,
        }

    result = retriever.invoke(query)
    context = [doc.page_content for doc in result]
    metadata = [doc.metadata for doc in result]

    return {
        "query": query,
        "context": context,
        "metadata": metadata,
        "source_file": _THREAD_METADATA.get(str(thread_id), {}).get("filename"),
    }

tools = [search_tool, get_stock_price, calculator, rag_tool]
llm_with_tools = llm.bind_tools(tools)

# -------------------
# 4. State
# -------------------
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# -------------------
# 5. Nodes
# -------------------
def chat_node(state: ChatState, config=None):
    """LLM node that may answer or request a tool call."""
    thread_id = None
    if config and isinstance(config, dict):
        thread_id = config.get("configurable", {}).get("thread_id")

    system_message = SystemMessage(
        content=(
            "You are a helpful assistant. For questions about the uploaded PDF, call "
            "the `rag_tool` and include the thread_id "
            f"`{thread_id}`. You can also use the web search, stock price, and "
            "calculator tools when helpful. If no document is available, ask the user "
            "to upload a PDF."
        )
    )

    messages = [system_message, *state["messages"]]
    response = llm_with_tools.invoke(messages, config=config)
    return {"messages": [response]}

tool_node = ToolNode(tools)

# -------------------
# 6. Checkpointer
# -------------------
conn = sqlite3.connect(database="chatbot.db", check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)

# Create metadata table
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

# -------------------
# 7. Graph
# -------------------
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")

chatbot = graph.compile(checkpointer=checkpointer)

# -------------------
# 8. Helpers
# -------------------
def retrieve_all_threads():
    cursor = conn.cursor()
    cursor.execute("""
        SELECT thread_id FROM thread_metadata 
        WHERE is_deleted = 0
        ORDER BY created_at DESC
    """)
    rows = cursor.fetchall()
    return [row[0] for row in rows]

def thread_has_document(thread_id: str) -> bool:
    return str(thread_id) in _THREAD_RETRIEVERS

def thread_document_metadata(thread_id: str) -> dict:
    return _THREAD_METADATA.get(str(thread_id), {})

def get_thread_display_name(thread_id: str) -> str:
    cursor = conn.cursor()
    cursor.execute("SELECT display_name FROM thread_metadata WHERE thread_id = ?", (str(thread_id),))
    row = cursor.fetchone()
    return row[0] if row and row[0] else str(thread_id)

def set_thread_display_name(thread_id: str, display_name: str):
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO thread_metadata (thread_id, display_name)
        VALUES (?, ?)
        ON CONFLICT(thread_id) DO UPDATE SET display_name = ?
    """, (str(thread_id), display_name, display_name))
    conn.commit()

def auto_name_thread_from_first_message(thread_id: str, first_message: str):
    if len(first_message) <= 20:
        display_name = first_message
    else:
        display_name = first_message[:20] + "..."
    set_thread_display_name(thread_id, display_name)

def delete_thread(thread_id: str):
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO thread_metadata (thread_id, is_deleted)
        VALUES (?, 1)
        ON CONFLICT(thread_id) DO UPDATE SET is_deleted = 1
    """, (str(thread_id),))
    conn.commit()

def rename_thread(thread_id: str, new_name: str):
    set_thread_display_name(thread_id, new_name)

def generate_share_token(thread_id: str) -> str:
    cursor = conn.cursor()
    cursor.execute("SELECT share_token FROM thread_metadata WHERE thread_id = ?", (str(thread_id),))
    row = cursor.fetchone()
    
    if row and row[0]:
        return row[0]
    
    token = secrets.token_urlsafe(16)
    cursor.execute("""
        INSERT INTO thread_metadata (thread_id, share_token)
        VALUES (?, ?)
        ON CONFLICT(thread_id) DO UPDATE SET share_token = ?
    """, (str(thread_id), token, token))
    conn.commit()
    return token

def export_thread_conversation(thread_id: str) -> dict:
    state = chatbot.get_state(config={'configurable': {'thread_id': str(thread_id)}})
    messages = state.values.get('messages', [])
    
    conversation = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            role = 'user'
        elif not isinstance(msg, SystemMessage):
            role = 'assistant'
        else:
            continue
        conversation.append({'role': role, 'content': msg.content})
    
    return {
        'thread_id': str(thread_id),
        'display_name': get_thread_display_name(thread_id),
        'conversation': conversation,
        'has_document': thread_has_document(thread_id),
        'document_info': thread_document_metadata(thread_id)
    }

def ensure_thread_exists(thread_id: str):
    cursor = conn.cursor()
    cursor.execute("""
        INSERT OR IGNORE INTO thread_metadata (thread_id, is_deleted)
        VALUES (?, 0)
    """, (str(thread_id),))
    conn.commit()