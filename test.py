import uuid
from graphs.chat_workflow import graph as chat_bot

config = {"configurable": {"thread_id": str(uuid.uuid4())}}

inputs = {"input": "Hello My name is JOHN what can you do?"}
result = chat_bot.invoke(inputs, config=config)
print(result)

inputs = {"input": "What's my name and what's yours?"}
result = chat_bot.invoke(inputs, config=config)
print(result)
