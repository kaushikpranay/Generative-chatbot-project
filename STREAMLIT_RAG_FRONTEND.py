import streamlit as st
from langgraph_rag_backend import (
    chatbot,
    retrieve_all_threads,
    get_thread_display_name,
    rename_thread,
    delete_thread,
    generate_share_token,
    export_thread_conversation,
    auto_name_thread_from_first_message,
    ensure_thread_exists,
    ingest_pdf,
    thread_has_document,
    thread_document_metadata
)
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
import uuid
import json

# =========================== Utilities ===========================
def generate_thread_id():
    return uuid.uuid4()

def reset_chat():
    thread_id = generate_thread_id()
    st.session_state["thread_id"] = thread_id
    add_thread(thread_id)
    st.session_state["message_history"] = []
    st.session_state["is_first_message"] = True

def add_thread(thread_id):
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)
    ensure_thread_exists(str(thread_id))

def load_conversation(thread_id):
    state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
    return state.values.get("messages", [])

# ======================= Session Initialization ===================
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = retrieve_all_threads()

if "is_first_message" not in st.session_state:
    st.session_state["is_first_message"] = True

if "rename_mode" not in st.session_state:
    st.session_state["rename_mode"] = {}

add_thread(st.session_state["thread_id"])

# ============================ Sidebar ============================
st.sidebar.title("LangGraph RAG Chatbot")

if st.sidebar.button("New Chat"):
    reset_chat()
    st.rerun()

# PDF Upload Section
st.sidebar.header("📄 Upload PDF")
uploaded_file = st.sidebar.file_uploader("Upload a PDF for this chat", type=["pdf"])

if uploaded_file:
    if st.sidebar.button("Process PDF"):
        with st.sidebar.spinner("Processing PDF..."):
            file_bytes = uploaded_file.read()
            result = ingest_pdf(
                file_bytes, 
                str(st.session_state["thread_id"]), 
                uploaded_file.name
            )
            st.sidebar.success(f"✅ Processed: {result['filename']}")
            st.sidebar.info(f"📊 {result['documents']} pages, {result['chunks']} chunks")

# Show current document status
if thread_has_document(str(st.session_state["thread_id"])):
    doc_info = thread_document_metadata(str(st.session_state["thread_id"]))
    st.sidebar.success(f"📑 Active: {doc_info.get('filename', 'Unknown')}")

st.sidebar.header("My Conversations")

# Refresh threads list
st.session_state["chat_threads"] = retrieve_all_threads()

for thread_id in st.session_state["chat_threads"]:
    thread_id_str = str(thread_id)
    display_name = get_thread_display_name(thread_id_str)
    
    col1, col2, col3, col4 = st.sidebar.columns([3, 1, 1, 1])
    
    # Show thread button or rename input
    with col1:
        if st.session_state["rename_mode"].get(thread_id_str, False):
            new_name = st.text_input(
                "Rename",
                value=display_name,
                key=f"rename_input_{thread_id_str}",
                label_visibility="collapsed"
            )
            if st.button("✓", key=f"save_rename_{thread_id_str}"):
                rename_thread(thread_id_str, new_name)
                st.session_state["rename_mode"][thread_id_str] = False
                st.rerun()
        else:
            # Add PDF indicator if thread has document
            prefix = "📄 " if thread_has_document(thread_id_str) else ""
            if st.button(
                f"{prefix}{display_name}", 
                key=f"thread_{thread_id_str}", 
                use_container_width=True
            ):
                st.session_state["thread_id"] = thread_id
                messages = load_conversation(thread_id)
                
                temp_messages = []
                for msg in messages:
                    if isinstance(msg, HumanMessage):
                        role = "user"
                        temp_messages.append({"role": role, "content": msg.content})
                    elif isinstance(msg, AIMessage):
                        role = "assistant"
                        temp_messages.append({"role": role, "content": msg.content})
                
                st.session_state["message_history"] = temp_messages
                st.session_state["is_first_message"] = False
                st.rerun()
    
    # Rename button
    with col2:
        if st.button("✏️", key=f"rename_{thread_id_str}"):
            st.session_state["rename_mode"][thread_id_str] = True
            st.rerun()
    
    # Share button
    with col3:
        if st.button("🔗", key=f"share_{thread_id_str}"):
            share_token = generate_share_token(thread_id_str)
            conversation_data = export_thread_conversation(thread_id_str)
            
            st.sidebar.success(f"Share Token: {share_token}")
            st.sidebar.download_button(
                label="📥 Download",
                data=json.dumps(conversation_data, indent=2),
                file_name=f"chat_{thread_id_str[:8]}.json",
                mime="application/json",
                key=f"download_{thread_id_str}"
            )
    
    # Delete button
    with col4:
        if st.button("🗑️", key=f"delete_{thread_id_str}"):
            delete_thread(thread_id_str)
            if str(st.session_state["thread_id"]) == thread_id_str:
                reset_chat()
            st.rerun()

# ============================ Main UI ============================

# Render history
for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.text(message["content"])

user_input = st.chat_input("Type here")

if user_input:
    # Auto-name thread on first message
    if st.session_state["is_first_message"]:
        auto_name_thread_from_first_message(str(st.session_state["thread_id"]), user_input)
        st.session_state["is_first_message"] = False
    
    # Show user's message
    st.session_state["message_history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.text(user_input)

    CONFIG = {
        "configurable": {"thread_id": str(st.session_state["thread_id"])},
        "metadata": {"thread_id": str(st.session_state["thread_id"])},
        "run_name": "chat_turn",
    }

    # Assistant streaming block
    with st.chat_message("assistant"):
        status_holder = {"box": None}

        def ai_only_stream():
            for message_chunk, metadata in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages",
            ):
                # Lazily create & update the SAME status container when any tool runs
                if isinstance(message_chunk, ToolMessage):
                    tool_name = getattr(message_chunk, "name", "tool")
                    if status_holder["box"] is None:
                        status_holder["box"] = st.status(
                            f"🔧 Using `{tool_name}` …", expanded=True
                        )
                    else:
                        status_holder["box"].update(
                            label=f"🔧 Using `{tool_name}` …",
                            state="running",
                            expanded=True,
                        )

                # Stream ONLY assistant tokens
                if isinstance(message_chunk, AIMessage):
                    yield message_chunk.content

        ai_message = st.write_stream(ai_only_stream())

        # Finalize only if a tool was actually used
        if status_holder["box"] is not None:
            status_holder["box"].update(
                label="✅ Tool finished", state="complete", expanded=False
            )

    # Save assistant message
    st.session_state["message_history"].append(
        {"role": "assistant", "content": ai_message}
    )