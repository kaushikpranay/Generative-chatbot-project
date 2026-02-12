import os
from pydantic.v1.fields import FieldInfo as FieldInfoV1
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

load_dotenv()

# This will automatically generate or give a view on of log on Langsmith
llm = ChatOllama(model="gpt-oss:20b")
response = llm.invoke([HumanMessage(content="Hello, Langsmith!")])
print(response.content)
print("\nCheck Langsmith dashboard for trace!")