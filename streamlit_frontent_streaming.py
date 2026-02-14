#streamlit_frontend_streaming.py

import streamlit as st
from langgraph_backend import chatbot
from langchain_core.messages import HumanMessage
from typing import Any, cast

#st.session_state -> dict ->
CONFIG = {'configurable': {'thread_id': 'thread-1'}}

if 'message_history' not in st.session_state:
    st.session_state['message_history']=[]


#loading the conversation history
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.text(message['content'])

user_input = st.chat_input("Type here")

if user_input:
    st.session_state['message_history'].append({'role':'user', 'content': user_input})
    with st.chat_message('user'):
        st.text(user_input)

    with st.chat_message('assistant'):
        def _stream_gen():
            for message_chunk, metadata in chatbot.stream(
                {'messages': [HumanMessage(content=user_input)]},
                config=cast(Any, {'configurable': {'thread_id': 'thread-1'}}),
                stream_mode='messages'
            ):
                yield getattr(message_chunk, 'content', message_chunk)

        ai_message = st.write_stream(_stream_gen())

    st.session_state['message_history'].append({'role':'assistant', 'content': ai_message})