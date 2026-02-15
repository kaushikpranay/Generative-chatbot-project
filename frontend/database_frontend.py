#database_frontend.py

import streamlit as st
from backend.langgraph_database import (
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
from typing import Any, cast
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
    state = chatbot.get_state(config=cast(Any, {'configurable': {'thread_id': thread_id}}))
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

if 'show_menu' not in st.session_state:
    st.session_state['show_menu'] = None

if 'rename_thread_id' not in st.session_state:
    st.session_state['rename_thread_id'] = None

add_thread(st.session_state['thread_id'])

#==============================Sidebar UI==================================

st.sidebar.title('LangGraph Chatbot')

if st.sidebar.button('➕ New Chat', use_container_width=True):
    reset_chat()
    st.rerun()

st.sidebar.header('My Conversations')

# Refresh threads
st.session_state['chat_threads'] = retrieve_all_threads()

# Handle rename dialog
if st.session_state['rename_thread_id']:
    thread_id_str = st.session_state['rename_thread_id']
    current_name = get_thread_display_name(thread_id_str)
    
    with st.sidebar.form(key=f"rename_form_{thread_id_str}"):
        new_name = st.text_input("New name:", value=current_name)
        col1, col2 = st.columns(2)
        with col1:
            if st.form_submit_button("Save"):
                rename_thread(thread_id_str, new_name)
                st.session_state['rename_thread_id'] = None
                st.rerun()
        with col2:
            if st.form_submit_button("Cancel"):
                st.session_state['rename_thread_id'] = None
                st.rerun()

# Thread list
for thread_id in st.session_state['chat_threads']:
    thread_id_str = str(thread_id)
    display_name = get_thread_display_name(thread_id_str)
    
    col1, col2 = st.sidebar.columns([5, 1])
    
    with col1:
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
    
    with col2:
        if st.button("⋮", key=f"menu_{thread_id_str}"):
            st.session_state['show_menu'] = thread_id_str
            st.rerun()
    
    # Show menu if this thread's menu is active
    if st.session_state['show_menu'] == thread_id_str:
        with st.sidebar.container():
            st.markdown("---")
            
            if st.button("✏️ Rename", key=f"rename_{thread_id_str}", use_container_width=True):
                st.session_state['rename_thread_id'] = thread_id_str
                st.session_state['show_menu'] = None
                st.rerun()
            
            if st.button("🔗 Share", key=f"share_{thread_id_str}", use_container_width=True):
                share_token = generate_share_token(thread_id_str)
                conversation_data = export_thread_conversation(thread_id_str)
                
                st.success(f"Token: {share_token}")
                st.download_button(
                    label="📥 Download JSON",
                    data=json.dumps(conversation_data, indent=2),
                    file_name=f"chat_{thread_id_str[:8]}.json",
                    mime="application/json",
                    key=f"download_{thread_id_str}"
                )
                st.session_state['show_menu'] = None
            
            if st.button("🗑️ Delete", key=f"delete_{thread_id_str}", use_container_width=True, type="secondary"):
                delete_thread(thread_id_str)
                if str(st.session_state['thread_id']) == thread_id_str:
                    reset_chat()
                st.session_state['show_menu'] = None
                st.rerun()
            
            if st.button("✖️ Close", key=f"close_menu_{thread_id_str}", use_container_width=True):
                st.session_state['show_menu'] = None
                st.rerun()
            
            st.markdown("---")

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
        def _stream_gen():
            for message_chunk, metadata in chatbot.stream(
                {'messages': [HumanMessage(content=user_input)]},
                config=cast(Any, CONFIG),
                stream_mode='messages'
            ):
                yield getattr(message_chunk, 'content', message_chunk)

        ai_message = st.write_stream(_stream_gen())

    st.session_state['message_history'].append({'role': 'assistant', 'content': ai_message})