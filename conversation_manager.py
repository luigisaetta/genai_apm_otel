"""
Conversation Manager
"""

from typing import List, Dict
from langchain_core.messages import BaseMessage


class ConversationManager:
    """
    To handle the conversation history
    """

    def __init__(self, max_messages: int = 100):
        # Dictionary to store conversation history, where key is conv_id
        self._conversations: Dict[str, List[BaseMessage]] = {}
        self._max_messages = max_messages

    def get_conversation(self, conv_id: str) -> List[BaseMessage]:
        """Retrieve the conversation history for a given conversation ID."""
        return self._conversations.get(conv_id, [])

    def add_message(self, conv_id: str, message: BaseMessage):
        """Add a message to the conversation history for a given conversation ID."""
        if conv_id not in self._conversations:
            self._conversations[conv_id] = []
        self._conversations[conv_id].append(message)

        # Remove oldest messages if the limit is exceeded
        while len(self._conversations[conv_id]) > self._max_messages:
            self._conversations[conv_id].pop(0)

    def clear_conversation(self, conv_id: str):
        """Clear the conversation history for a given conversation ID."""
        if conv_id in self._conversations:
            del self._conversations[conv_id]
