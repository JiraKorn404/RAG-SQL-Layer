"""Chat and query history, stored in the `chat_memory` schema through the chat role.

chat.py holds the conversations (turns and their metrics), queries.py the successful past queries
used as SQL examples. They are the only modules that touch `chat_memory`.
"""
