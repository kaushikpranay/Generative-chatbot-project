# LangGraph Chatbot Project - Complete Interview Prep Guide

## PROJECT OVERVIEW

**Project Name:** LangGraph Chatbot with Ollama Integration
**Tech Stack:** Python, LangGraph, Streamlit, Ollama, SQLite
**Purpose:** Conversational AI chatbot with memory, tools, and RAG capabilities
**Architecture:** State-based graph workflow with persistent storage

---

## 1. CORE LIBRARIES & FRAMEWORKS

### LangGraph (v0.2.0+)
**What it is:** Orchestration framework for building stateful, multi-actor applications with LLMs

**Key Concepts You MUST Know:**
- **StateGraph:** Directed graph where nodes are functions and edges define flow
- **State:** Shared data structure passed between nodes (type-safe with TypedDict)
- **Checkpointing:** Saves state at each step for resumability and time-travel
- **Memory:** Persistent conversation history across sessions

**Core Classes:**
```python
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import ToolNode, tools_condition
```

**When Recruiter Asks: "Why LangGraph over LangChain?"**
Answer:
- LangGraph = LangChain + state management + graph orchestration
- Better for complex multi-step workflows
- Built-in memory/checkpointing
- More control over agent behavior
- LangChain is for simple chains; LangGraph is for agentic systems

### LangChain (v0.3.0+)
**What it is:** Framework for developing LLM-powered applications

**Key Components Used:**
```python
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
```

**Message Types:**
- **HumanMessage:** User input
- **AIMessage:** Bot response
- **SystemMessage:** Instructions/context (not shown to user)
- **ToolMessage:** Tool execution results

**When Recruiter Asks: "Explain the message flow"**
Answer:
1. User types → HumanMessage created
2. LLM processes messages → AIMessage returned
3. If tool needed → ToolMessage with results
4. All messages stored in state["messages"] list
5. Full history passed to LLM each time (context window)

### Streamlit (v1.40.0+)
**What it is:** Python framework for building web UIs quickly

**Key Features Used:**
```python
import streamlit as st

# Session state (persists across reruns)
st.session_state.messages = []
st.session_state.thread_id = "abc123"

# UI Components
st.title("Chatbot")
st.sidebar.selectbox("Conversations", options)
st.chat_input("Type message...")
st.chat_message("user").write("Hello")

# Streaming
with st.chat_message("assistant"):
    with st.spinner("Thinking..."):
        response = stream_response()
```

**When Recruiter Asks: "How does Streamlit handle state?"**
Answer:
- Reruns entire script on every interaction
- `st.session_state` persists data between reruns
- No traditional backend - everything in Python
- Great for prototypes, not ideal for production at scale

### Ollama
**What it is:** Run large language models locally (like ChatGPT, but on your machine)

**Key Points:**
- FREE alternative to OpenAI API
- Privacy - data never leaves your machine
- Models: llama3.2, mixtral, mistral, etc.
- Runs via HTTP API (localhost:11434)

**Integration:**
```python
from langchain_community.chat_models import ChatOllama

llm = ChatOllama(
    model="llama3.2",
    temperature=0.7,
    num_ctx=4096  # Context window size
)
```

**When Recruiter Asks: "Why Ollama vs OpenAI?"**
Answer:
- **Cost:** $0 vs $0.06/message (GPT-4)
- **Privacy:** Local vs cloud
- **Latency:** Depends on hardware vs ~500ms
- **Quality:** llama3.2 ≈ GPT-3.5, mixtral ≈ GPT-4
- **Use case:** Development/personal vs production

---

## 2. PROJECT ARCHITECTURE

### State Management
```python
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages

class State(TypedDict):
    messages: Annotated[list, add_messages]  # Auto-appends messages
```

**Why TypedDict?**
- Type safety
- IDE autocomplete
- Runtime validation

**What is `add_messages` reducer?**
- Custom function for list merging
- Prevents duplicate messages
- Handles message updates by ID

### Graph Structure (Basic Chatbot)
```
START → chatbot_node → END
```

**chatbot_node function:**
```python
def chatbot_node(state: State):
    response = llm.invoke(state["messages"])
    return {"messages": [response]}
```

**Graph Building:**
```python
graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot_node)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)
```

### Graph Structure (With Tools)
```
START → chatbot → [conditional]
                     ↓
                  has tool_calls? → tools → chatbot (loop)
                     ↓
                  no tool_calls? → END
```

**Conditional Routing:**
```python
from langgraph.prebuilt import tools_condition

graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,  # Checks if AIMessage has tool_calls
)
graph_builder.add_edge("tools", "chatbot")
```

**When Recruiter Asks: "How does tool calling work?"**
Answer:
1. LLM decides if it needs a tool (returns tool_calls in AIMessage)
2. tools_condition routes to ToolNode
3. ToolNode executes tool, returns ToolMessage
4. Flow returns to chatbot with tool results
5. LLM generates final response with tool output

---

## 3. PERSISTENCE & MEMORY

### SQLite Checkpointing
```python
from langgraph.checkpoint.sqlite import SqliteSaver

memory = SqliteSaver.from_conn_string("chatbot.db")
graph = graph_builder.compile(checkpointer=memory)
```

**What gets stored:**
- Full state at each step
- Message history
- Tool call results
- Thread metadata

**Database Schema (simplified):**
```sql
checkpoints (
    thread_id TEXT,
    checkpoint_id TEXT,
    parent_id TEXT,
    state BLOB,  -- Pickled Python object
    created_at TIMESTAMP
)
```

### Thread Management
```python
config = {
    "configurable": {
        "thread_id": "user-123-conv-1"
    }
}

# Invoke with thread
result = graph.invoke(state, config=config)

# Retrieve history
for state in graph.get_state_history(config):
    print(state.values["messages"])
```

**Thread ID Strategy:**
- UUID for unique conversations
- Format: `user_id-conversation_id` or just UUID
- Stored in session state (Streamlit)

**When Recruiter Asks: "How do you handle concurrent users?"**
Answer:
- Each user gets unique thread_id
- SQLite handles concurrent reads (not writes)
- For production: Use PostgreSQL with AsyncSaver
- Session state isolates user data in Streamlit

---

## 4. TOOL INTEGRATION

### Defining Tools
```python
from langchain_core.tools import tool

@tool
def search(query: str) -> str:
    """Search the web for information."""
    from duckduckgo_search import DDGS
    results = DDGS().text(query, max_results=5)
    return str(results)

@tool
def calculator(expression: str) -> str:
    """Evaluate mathematical expressions."""
    try:
        return str(eval(expression))
    except:
        return "Invalid expression"
```

**Tool Decorator Features:**
- Generates JSON schema automatically
- Docstring becomes tool description (LLM sees this!)
- Type hints define parameters

### Tool Node
```python
from langgraph.prebuilt import ToolNode

tools = [search, calculator]
tool_node = ToolNode(tools)

graph_builder.add_node("tools", tool_node)
```

**What ToolNode does:**
1. Receives AIMessage with tool_calls
2. Executes each tool
3. Returns ToolMessage with results
4. Handles errors gracefully

### DuckDuckGo Search
```python
from duckduckgo_search import DDGS

ddgs = DDGS()
results = ddgs.text("Python tutorials", max_results=5)
# Returns: [{"title": "...", "href": "...", "body": "..."}, ...]
```

**When Recruiter Asks: "Why DuckDuckGo?"**
Answer:
- No API key required
- Free unlimited searches
- Privacy-focused
- Good for demos/learning
- Production: Use Google Search API, Bing API, or Tavily

---

## 5. STREAMING RESPONSES

### Basic Streaming
```python
for chunk in graph.stream(state, config):
    print(chunk)
```

**Output format:**
```python
{"chatbot": {"messages": [AIMessage(content="Hello")]}}
{"tools": {"messages": [ToolMessage(content="...")]}}
{"chatbot": {"messages": [AIMessage(content="Final answer")]}}
```

### Token-Level Streaming
```python
for event in graph.stream_events(state, config, version="v2"):
    if event["event"] == "on_chat_model_stream":
        token = event["data"]["chunk"].content
        print(token, end="", flush=True)
```

**Streamlit Implementation:**
```python
with st.chat_message("assistant"):
    message_placeholder = st.empty()
    full_response = ""
    
    for chunk in graph.stream(state, config):
        if "chatbot" in chunk:
            full_response += chunk["chatbot"]["messages"][-1].content
            message_placeholder.markdown(full_response + "▌")
    
    message_placeholder.markdown(full_response)
```

---

## 6. RAG IMPLEMENTATION (If you build it)

### Vector Store Setup
```python
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 1. Load documents
from langchain_community.document_loaders import TextLoader
loader = TextLoader("docs.txt")
documents = loader.load()

# 2. Split into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = splitter.split_documents(documents)

# 3. Create embeddings & vector store
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(chunks, embeddings)
```

### Retrieval Node
```python
def retrieval_node(state: State):
    query = state["messages"][-1].content
    docs = vectorstore.similarity_search(query, k=3)
    context = "\n\n".join([doc.page_content for doc in docs])
    
    # Inject context into system message
    system_msg = SystemMessage(content=f"Context:\n{context}")
    return {"messages": [system_msg]}
```

**Graph Structure:**
```
START → retrieval → chatbot → END
```

**When Recruiter Asks: "Explain your RAG pipeline"**
Answer:
1. **Indexing:** Documents → chunks → embeddings → vector store
2. **Retrieval:** User query → embedding → similarity search → top K docs
3. **Augmentation:** Inject docs as context in prompt
4. **Generation:** LLM generates answer with context

**Chunk Size Trade-offs:**
- Small (200-500): Precise retrieval, less context
- Large (1000-2000): More context, less precise
- Overlap: Prevents cutting mid-sentence

---

## 7. LANGSMITH MONITORING

### Setup
```python
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "your-key"
os.environ["LANGCHAIN_PROJECT"] = "langgraph-chatbot"
```

**What gets traced:**
- Every LLM call (input/output/tokens)
- Tool executions
- Graph state transitions
- Latency metrics
- Error logs

### Custom Tracing
```python
from langsmith import traceable

@traceable(name="custom_function", run_type="chain")
def my_function(input):
    # Your code
    return output
```

### Feedback Collection
```python
from langsmith import Client

client = Client()
client.create_feedback(
    run_id=run_id,
    key="user_rating",
    score=1.0,  # 0.0 to 1.0
    comment="Great response!"
)
```

**When Recruiter Asks: "How do you monitor production?"**
Answer:
- LangSmith for LLM traces & costs
- Streamlit analytics for user behavior
- Token counting for cost tracking
- Error rates & latency in LangSmith
- User feedback (thumbs up/down)

---

## 8. MCP INTEGRATION (Model Context Protocol)

**What is MCP?**
- Standard protocol for connecting LLMs to external data sources
- Think "API for AI assistants"
- Enables reading files, databases, APIs, etc.

**Example MCP Server:**
```python
from mcp import Server, Resource

server = Server("my-app")

@server.resource("file://documents/{path}")
async def get_document(path: str):
    with open(path) as f:
        return f.read()
```

**Integration with LangGraph:**
```python
from langchain_mcp import MCPClient

mcp = MCPClient("http://localhost:3000")

@tool
def read_document(path: str) -> str:
    """Read a document via MCP."""
    return mcp.get_resource(f"file://documents/{path}")
```

---

## 9. COMMON RECRUITER QUESTIONS

### Q1: "Walk me through the flow when a user sends a message"

**Answer:**
1. User types message in Streamlit input
2. HumanMessage created, added to state["messages"]
3. State passed to graph.invoke(state, config)
4. Graph routes to chatbot_node
5. LLM receives full message history
6. If tool needed: routes to tools → executes → back to chatbot
7. Final AIMessage returned
8. State saved to SQLite via checkpointer
9. Response displayed in Streamlit
10. UI reruns, message added to chat history

### Q2: "How do you handle errors?"

**Honest Answer:**
Current implementation has NO error handling. In production, I'd add:

```python
def chatbot_node(state: State):
    try:
        response = llm.invoke(state["messages"])
        return {"messages": [response]}
    except Exception as e:
        error_msg = AIMessage(content="Sorry, I encountered an error.")
        logger.error(f"LLM error: {e}")
        return {"messages": [error_msg]}
```

**Additional safeguards:**
- Retry logic with exponential backoff
- Fallback to simpler model
- Circuit breaker for repeated failures
- User-friendly error messages

### Q3: "How would you scale this for 1000 concurrent users?"

**Answer:**
Current setup won't scale. Here's what I'd change:

1. **Database:** SQLite → PostgreSQL with connection pooling
2. **Backend:** Streamlit → FastAPI + WebSockets
3. **LLM:** Ollama → OpenAI API with rate limiting
4. **State:** In-memory → Redis for session cache
5. **Deployment:** Docker + Kubernetes for horizontal scaling
6. **Monitoring:** Add Prometheus, Grafana
7. **Queue:** Celery for async tool execution

**Architecture:**
```
Load Balancer
    ↓
FastAPI (multiple instances)
    ↓
Redis (session store)
    ↓
PostgreSQL (conversation history)
    ↓
OpenAI API (with caching layer)
```

### Q4: "Explain the difference between invoke() and stream()"

**Answer:**
```python
# invoke() - waits for full response
result = graph.invoke(state, config)
# Returns: {"messages": [AIMessage(content="full response")]}

# stream() - yields intermediate states
for chunk in graph.stream(state, config):
    print(chunk)
# Yields: 
# {"chatbot": {"messages": [AIMessage(...)]}}
# {"tools": {"messages": [ToolMessage(...)]}}
# ...
```

**Use invoke when:** You need the complete result at once (API endpoints)
**Use stream when:** Real-time UI updates (chat interfaces)

### Q5: "What are the limitations of this approach?"

**Honest Answer:**
1. **Context Window:** Limited to ~4K tokens (older models) or 128K (newer)
2. **Latency:** Multiple LLM calls in tool loop (3-5 seconds)
3. **Cost:** Token usage grows with conversation length
4. **Hallucinations:** LLM may invent facts without RAG
5. **Tool Reliability:** Dependent on external APIs
6. **No Authentication:** Anyone can access
7. **SQLite Bottleneck:** Write-heavy workloads will fail

### Q6: "How do you ensure data privacy?"

**Current State:** Ollama keeps data local, but no user isolation

**Production Answer:**
1. **User Authentication:** JWT tokens, OAuth
2. **Thread Isolation:** thread_id = f"{user_id}-{uuid}"
3. **Data Encryption:** At rest (database) and in transit (HTTPS)
4. **Access Control:** Row-level security in PostgreSQL
5. **Audit Logs:** Track who accessed what
6. **Data Retention:** Auto-delete after 30 days
7. **GDPR Compliance:** Right to deletion, data export

### Q7: "How would you test this?"

**Unit Tests:**
```python
def test_chatbot_node():
    state = {"messages": [HumanMessage(content="Hi")]}
    result = chatbot_node(state)
    assert len(result["messages"]) == 1
    assert isinstance(result["messages"][0], AIMessage)
```

**Integration Tests:**
```python
def test_graph_flow():
    graph = build_graph()
    state = {"messages": [HumanMessage(content="What is 2+2?")]}
    result = graph.invoke(state, config={"thread_id": "test"})
    assert "4" in result["messages"][-1].content
```

**E2E Tests:**
- Selenium for Streamlit UI
- Test full conversation flows
- Tool invocation scenarios

### Q8: "What's your token optimization strategy?"

**Answer:**
1. **Summarization:** Compress old messages after 10 turns
2. **Sliding Window:** Keep only last N messages
3. **Semantic Pruning:** Remove less relevant messages
4. **Prompt Caching:** Reuse system prompts (OpenAI feature)
5. **Model Selection:** Use cheaper models for simple queries

```python
def prune_messages(messages, max_tokens=3000):
    from tiktoken import encoding_for_model
    enc = encoding_for_model("gpt-4")
    
    total_tokens = sum(len(enc.encode(m.content)) for m in messages)
    
    if total_tokens < max_tokens:
        return messages
    
    # Keep system + last 10 messages
    return [messages[0]] + messages[-10:]
```

---

## 10. TECHNICAL DEEP DIVES

### How Checkpointing Works Internally

1. **Before execution:**
   - SqliteSaver loads latest state for thread_id
   - State passed to first node

2. **After each node:**
   - New state created by merging node output
   - State serialized (pickle)
   - Saved to database with checkpoint_id
   - Linked to parent checkpoint

3. **On resume:**
   - Load state by thread_id + checkpoint_id
   - Continue from that point

**Database Operations:**
```sql
-- Save checkpoint
INSERT INTO checkpoints (thread_id, checkpoint_id, parent_id, state)
VALUES (?, ?, ?, ?);

-- Load latest
SELECT state FROM checkpoints
WHERE thread_id = ?
ORDER BY created_at DESC
LIMIT 1;

-- Time travel
SELECT state FROM checkpoints
WHERE thread_id = ? AND checkpoint_id = ?;
```

### How Tool Calling Works (OpenAI Format)

**LLM Output (with tool call):**
```json
{
  "role": "assistant",
  "content": null,
  "tool_calls": [
    {
      "id": "call_abc123",
      "type": "function",
      "function": {
        "name": "search",
        "arguments": "{\"query\": \"Python tutorials\"}"
      }
    }
  ]
}
```

**ToolNode Execution:**
```python
# 1. Extract tool call
tool_call = ai_message.tool_calls[0]
tool_name = tool_call["name"]
tool_args = json.loads(tool_call["args"])

# 2. Execute tool
tool_func = tools_by_name[tool_name]
result = tool_func(**tool_args)

# 3. Create ToolMessage
tool_message = ToolMessage(
    content=result,
    tool_call_id=tool_call["id"]
)
```

**Back to LLM:**
```python
messages = [
    HumanMessage("Search for Python tutorials"),
    AIMessage(tool_calls=[...]),
    ToolMessage(content="[search results]", tool_call_id="call_abc123")
]
final_response = llm.invoke(messages)
# LLM now generates response using search results
```

### State Reducers Explained

**Default Behavior (without reducer):**
```python
state["messages"] = new_messages  # Replaces entire list
```

**With `add_messages` reducer:**
```python
from langgraph.graph.message import add_messages

class State(TypedDict):
    messages: Annotated[list, add_messages]

# Now:
state["messages"] = [new_msg]  # Appends to existing list
```

**How `add_messages` works:**
```python
def add_messages(existing: list, new: list) -> list:
    # Merge by message ID, append new messages
    by_id = {m.id: m for m in existing}
    for msg in new:
        if msg.id in by_id:
            by_id[msg.id] = msg  # Update
        else:
            by_id[msg.id] = msg  # Add
    return list(by_id.values())
```

---

## 11. GOTCHAS & EDGE CASES

### Issue 1: Context Window Overflow
**Problem:** Conversation gets too long, exceeds model limit
**Solution:** Implement message pruning or summarization

### Issue 2: Tool Loop Infinite Recursion
**Problem:** LLM keeps calling tools indefinitely
**Solution:**
```python
from langgraph.graph import END

def chatbot_node(state: State):
    if len(state["messages"]) > 50:  # Max turns
        return {"messages": [AIMessage(content="Let's start fresh.")]}
    # ... normal flow
```

### Issue 3: Concurrent Thread Writes (SQLite)
**Problem:** Two users save state simultaneously → database locked
**Solution:** Use PostgreSQL or implement retry logic

### Issue 4: Streaming Cuts Off Mid-Sentence
**Problem:** Network issue or timeout
**Solution:** Buffer tokens, send in chunks

### Issue 5: Tool Returns Huge JSON
**Problem:** Search returns 10MB of data, exceeds context window
**Solution:**
```python
@tool
def search(query: str) -> str:
    results = ddgs.text(query, max_results=5)
    # Truncate each result
    truncated = [r["body"][:200] for r in results]
    return "\n".join(truncated)
```

---

## 12. KEY METRICS TO TRACK

### Performance Metrics
- **Latency:** Time from user input to response
  - Target: <2s for simple queries, <5s with tools
- **Token Usage:** Tokens per conversation
  - Track: prompt_tokens + completion_tokens
- **Tool Success Rate:** % of tool calls that succeed
- **Error Rate:** % of requests that fail

### Business Metrics
- **Conversations per User:** Engagement metric
- **Average Conversation Length:** Quality metric
- **User Retention:** Do users come back?
- **Cost per Conversation:** Tokens × price per token

### Code to Track:
```python
from langsmith import Client
import time

start = time.time()
result = graph.invoke(state, config)
latency = time.time() - start

client = Client()
client.update_run(
    run_id=run_id,
    extra={
        "latency_seconds": latency,
        "message_count": len(state["messages"]),
        "thread_id": config["configurable"]["thread_id"]
    }
)
```

---

## 13. PRODUCTION CHECKLIST

- [ ] Replace SQLite with PostgreSQL
- [ ] Add user authentication (JWT/OAuth)
- [ ] Implement rate limiting
- [ ] Add comprehensive error handling
- [ ] Set up logging (structured JSON logs)
- [ ] Add monitoring (Prometheus/Grafana)
- [ ] Implement caching (Redis)
- [ ] Add unit/integration tests (pytest)
- [ ] Set up CI/CD pipeline
- [ ] Configure auto-scaling
- [ ] Add data encryption
- [ ] Implement backup strategy
- [ ] Add API documentation (OpenAPI/Swagger)
- [ ] Set up alerting (PagerDuty/Slack)
- [ ] Perform security audit
- [ ] Load testing (Locust/K6)
- [ ] GDPR compliance review
- [ ] Add feature flags
- [ ] Implement A/B testing
- [ ] Set up cost alerts

---

## 14. ADVANCED TOPICS (If They Go Deep)

### Multi-Agent Systems
**Scenario:** Different agents for different tasks

```python
from langgraph.graph import StateGraph

def researcher(state):
    # Searches web, reads docs
    pass

def writer(state):
    # Writes response
    pass

def critic(state):
    # Reviews quality
    pass

graph = StateGraph(State)
graph.add_node("researcher", researcher)
graph.add_node("writer", writer)
graph.add_node("critic", critic)

graph.add_edge("researcher", "writer")
graph.add_conditional_edges("critic", lambda x: "writer" if x["score"] < 0.8 else END)
```

### Human-in-the-Loop
**Scenario:** Require human approval for sensitive actions

```python
from langgraph.graph import interrupt

def sensitive_action(state):
    # Do something risky
    response = interrupt("Approve this action? (yes/no)")
    if response == "yes":
        # Proceed
        pass
    else:
        return {"error": "Action denied"}
```

### Parallel Execution
**Scenario:** Run multiple tools simultaneously

```python
from langgraph.graph import Send

def fan_out(state):
    return [
        Send("search", {"query": "topic1"}),
        Send("search", {"query": "topic2"}),
        Send("search", {"query": "topic3"}),
    ]

graph.add_conditional_edges("start", fan_out)
```

---

## 15. RAPID-FIRE Q&A

**Q: What's the difference between LangChain and LangGraph?**
A: LangChain = building blocks (chains, prompts). LangGraph = orchestration layer (graphs, state, memory).

**Q: Why use TypedDict over regular dict?**
A: Type safety, IDE support, runtime validation, better debugging.

**Q: Can you use multiple LLMs in one graph?**
A: Yes! Different nodes can use different models (GPT-4 for reasoning, GPT-3.5 for summarization).

**Q: How do you handle API rate limits?**
A: Exponential backoff, request queuing, fallback models, caching.

**Q: What's the max conversation length?**
A: Limited by context window (4K-128K tokens). Implement summarization for longer chats.

**Q: How do you prevent prompt injection?**
A: Input validation, system prompts with clear boundaries, output parsing, sandboxing.

**Q: Can you run this offline?**
A: Yes with Ollama! But tools (search) need internet.

**Q: What's your testing strategy?**
A: Unit (nodes), integration (graph flow), E2E (UI), load testing, LLM evals.

**Q: How do you version control prompts?**
A: Git for code, LangSmith for prompt experiments, feature flags for A/B tests.

**Q: What's your deployment process?**
A: Docker → GitHub Actions (CI) → K8s (staging) → manual approval → K8s (prod) → monitor.

---

## FINAL TIPS FOR INTERVIEW

1. **Don't oversell:** Admit what's missing (error handling, tests, auth)
2. **Show growth mindset:** "Current implementation is a POC. In production, I'd add..."
3. **Ask questions back:** "What scale are you handling? What's your current LLM stack?"
4. **Explain trade-offs:** "Ollama is great for development, but OpenAI for production quality"
5. **Draw diagrams:** Visualize the graph flow, architecture
6. **Know your numbers:** Tokens, latency, costs
7. **Be honest about limitations:** Context windows, hallucinations, tool reliability

**If they ask something you don't know:**
"I haven't implemented that yet, but here's how I'd approach it: [thoughtful answer]"

This shows problem-solving over memorization.

---

## BONUS: One-Sentence Summaries

**LangGraph:** Framework for building stateful, multi-step LLM applications with graph-based workflows
**Ollama:** Run LLMs locally for free without API keys
**Streamlit:** Python framework for building web UIs with minimal code
**Checkpointing:** Saving conversation state at each step for resumability and memory
**Tool Calling:** LLM decides when to use external functions (search, calculator, APIs)
**RAG:** Retrieve relevant documents, inject as context, generate informed responses
**LangSmith:** Observability platform for LLM applications (tracing, monitoring, debugging)
**MCP:** Standard protocol for LLMs to access external data sources
**State Reducers:** Functions that define how state updates are merged
**ToolNode:** Prebuilt component that executes tool calls and returns results

---

**YOU'RE READY.** Know this doc inside-out and you'll crush any technical deep-dive on this project.