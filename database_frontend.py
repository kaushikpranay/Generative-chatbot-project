import streamlit as st
from langgraph_database import (
    chatbot, 
    retrieve_all_threads,
    get_thread_display_name,
    rename_thread,
    delete_thread,
    generate_share_token,
    export_thread_conversation,
    auto_name_thread_from_first_message,
    ensure_thread_exists
)
from langchain_core.messages import HumanMessage
import uuid
import json

#=============================utility functions=============================

def generate_thread_id():
    thread_id = uuid.uuid4()
    return thread_id

def reset_chat():
    thread_id = generate_thread_id()
    st.session_state['thread_id'] = thread_id
    add_thread(st.session_state['thread_id'])
    st.session_state['message_history'] = []
    st.session_state['is_first_message'] = True

def add_thread(thread_id):
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(thread_id)
    ensure_thread_exists(str(thread_id))

def load_conversation(thread_id):
    state = chatbot.get_state(config={'configurable': {'thread_id': thread_id}})
    return state.values.get('messages', [])

#==========================session setup===========================

if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()

if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = retrieve_all_threads()

if 'is_first_message' not in st.session_state:
    st.session_state['is_first_message'] = True

if 'rename_mode' not in st.session_state:
    st.session_state['rename_mode'] = {}

add_thread(st.session_state['thread_id'])

#==============================Sidebar UI==================================

st.sidebar.title('LangGraph Chatbot')

if st.sidebar.button('New Chat'):
    reset_chat()
    st.rerun()

st.sidebar.header('My Conversations')

# Refresh threads list
st.session_state['chat_threads'] = retrieve_all_threads()

for thread_id in st.session_state['chat_threads']:
    thread_id_str = str(thread_id)
    display_name = get_thread_display_name(thread_id_str)
    
    col1, col2, col3, col4 = st.sidebar.columns([3, 1, 1, 1])
    
    # Show thread button or rename input
    with col1:
        if st.session_state['rename_mode'].get(thread_id_str, False):
            new_name = st.text_input(
                "Rename",
                value=display_name,
                key=f"rename_input_{thread_id_str}",
                label_visibility="collapsed"
            )
            if st.button("✓", key=f"save_rename_{thread_id_str}"):
                rename_thread(thread_id_str, new_name)
                st.session_state['rename_mode'][thread_id_str] = False
                st.rerun()
        else:
            if st.button(display_name, key=f"thread_{thread_id_str}", use_container_width=True):
                st.session_state['thread_id'] = thread_id
                messages = load_conversation(thread_id)
                
                temp_messages = []
                for msg in messages:
                    if isinstance(msg, HumanMessage):
                        role = 'user'
                    else:
                        role = 'assistant'
                    temp_messages.append({'role': role, 'content': msg.content})
                
                st.session_state['message_history'] = temp_messages
                st.session_state['is_first_message'] = False
                st.rerun()
    
    # Rename button
    with col2:
        if st.button("✏️", key=f"rename_{thread_id_str}"):
            st.session_state['rename_mode'][thread_id_str] = True
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
            if str(st.session_state['thread_id']) == thread_id_str:
                reset_chat()
            st.rerun()

#========================Main UI=================================

# Loading conversation history
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.text(message['content'])

user_input = st.chat_input('Type Here')

if user_input:
    # Auto-name thread on first message
    if st.session_state['is_first_message']:
        auto_name_thread_from_first_message(str(st.session_state['thread_id']), user_input)
        st.session_state['is_first_message'] = False
    
    # Add message to history
    st.session_state['message_history'].append({'role': 'user', 'content': user_input})
    with st.chat_message('user'):
        st.text(user_input)

    CONFIG = {
        "configurable": {"thread_id": str(st.session_state["thread_id"])},
        "metadata": {
            "thread_id": str(st.session_state["thread_id"])
        },
        "run_name": "chat_turn",
    }

    # Stream assistant response
    with st.chat_message('assistant'):
        ai_message = st.write_stream(
            message_chunk.content for message_chunk, metadata in chatbot.stream(
                {'messages': [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode='messages'
            )
        )

    st.session_state['message_history'].append({'role': 'assistant', 'content': ai_message})