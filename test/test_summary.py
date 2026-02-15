import sys
import os

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# NOW import
from backend.langgraph_backend import chatbot, ChatState
from langchain_core.messages import HumanMessage

CONFIG = {'configurable': {'thread_id': 'test-summary-thread'}}

test_messages = [
    "Tell me about Python",
    "What are decorators?",
    "How do I use async/await?",
    "Explain list comprehensions",
    "What's the difference between lists and tuples?",
    "How do I handle exceptions?",
    "What are context managers?",
    "Explain generators"
]

for msg in test_messages:
    response = chatbot.invoke(
        {'messages': [HumanMessage(content=msg)]},
        config=CONFIG
    )
    print(f"\n{'='*50}")
    print(f"USER: {msg}")
    print(f"BOT: {response['messages'][-1].content}")
    
    state = chatbot.get_state(config=CONFIG)
    if 'conversation_summary' in state.values:
        print(f"\n[BACKEND SUMMARY]:\n{state.values['conversation_summary']}")
# ```

## Fix 3: Nuclear Option (Create __init__.py files)

# Your structure should be:
# ```
# E:\Generative-chatbot-project\
# ├── backend\
# │   ├── __init__.py  ← ADD THIS
# │   ├── langgraph_backend.py
# │   └── ...
# ├── test\
# │   ├── __init__.py  ← ADD THIS
# │   └── test_summary.py
# └── frontend\
#     └── ...