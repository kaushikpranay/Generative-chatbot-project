# Quick test script
from langgraph_backend import chatbot
from langchain_core.messages import HumanMessage

state = {"messages": [HumanMessage(content="Hi i am kaushik")]}
result = chatbot.invoke(state, config={"configurable": {"thread_id": "test-1"}})
print(result["messages"][-1].content)