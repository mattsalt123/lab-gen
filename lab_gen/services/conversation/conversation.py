import os
import uuid

from typing import Any

from fastapi import FastAPI
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    BasePromptTemplate,
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    StringPromptTemplate,
)
from langchain_core.runnables import ConfigurableFieldSpec
from langchain_core.runnables.history import RunnableWithMessageHistory
from langfuse.callback import CallbackHandler

from lab_gen.datatypes.errors import NoConversationError
from lab_gen.datatypes.metadata import ConversationMetadata
from lab_gen.datatypes.models import ModelProvider
from lab_gen.services.conversation.block_content import (
    AzureBlockedContentTracker,
    BedrockBlockedContentTracker,
    BlockedContentTracker,
    VertexBlockedContentTracker,
)
from lab_gen.services.cosmos.cosmos_db import CosmosDBChatMessageHistory
from lab_gen.services.llm.lifetime import get_llm
from lab_gen.services.metrics.llm_metrics_counter import LLMMetricsCounter
from lab_gen.services.metrics.metrics import Metric


SYSTEM_MESSAGE = SystemMessage(
    content="You are a helpful AI bot, gifted at answering questions.",
)

class ConversationService:
    """Represents a conversation service."""

    def __init__(self, app: FastAPI, examples: dict[str, list[str]], prompts: dict[str, BasePromptTemplate]) -> None:
        self.app = app
        self.examples = examples
        self.prompts = prompts

    def create_chain(self, llm: BaseLanguageModel, prompt: ChatPromptTemplate) -> RunnableWithMessageHistory:
        """
        Create a chain to process a prompt and return a `RunnableWithMessageHistory` object.

        Args:
            llm (LLM): The LLM object used to generate the chain.
            prompt (Prompt): The prompt to be processed.

        Returns:
            RunnableWithMessageHistory: A `RunnableWithMessageHistory` object that represents the chain of operations.
        """
        str_chain = prompt | llm | StrOutputParser()
        return RunnableWithMessageHistory(
            str_chain,
            get_session_history=self.get_message_history,
            input_messages_key="input",
            history_messages_key="history",
            history_factory_config=[
                ConfigurableFieldSpec(
                    id="conversation_id",
                    annotation=str,
                    name="Conversation ID",
                    description="Unique identifier for a conversation.",
                    default="",
                    is_shared=True,
                ),
                ConfigurableFieldSpec(
                    id="user_id",
                    annotation=str,
                    name="User ID",
                    description="Unique identifier for a user.",
                    default="",
                    is_shared=True,
                ),
                ConfigurableFieldSpec(
                    id="metadata",
                    annotation=str,
                    name="Metadata",
                    description="Metadata about the conversation.",
                    default="",
                    is_shared=True,
                ),
            ],
        )

    def generate_config(self, meta: ConversationMetadata, conversation_id: str, llm: BaseLanguageModel) -> dict:
        """Generates configuration for a conversation.

        Args:
            meta: Metadata about the conversation.
            conversation_id: The conversation ID.
            llm: The language model used for the conversation.

        Returns:
            A dictionary containing the configuration for the conversation.
        """
        metrics_counter = LLMMetricsCounter(llm)
        match meta.provider:
            case ModelProvider.AZURE:
                blocked_content_counter = AzureBlockedContentTracker(llm)
            case ModelProvider.VERTEX:
                blocked_content_counter = VertexBlockedContentTracker(llm)
            case ModelProvider.BEDROCK:
                blocked_content_counter = BedrockBlockedContentTracker(llm)
            case _:
                blocked_content_counter = BlockedContentTracker(llm)

        callbacks = [metrics_counter, blocked_content_counter]

        if os.getenv("LANGFUSE_HOST"):
            langfuse_handler = CallbackHandler()
            langfuse_handler.user_id = meta.business_user
            langfuse_handler.session_id = conversation_id
            callbacks.append(langfuse_handler)

        return {
            "callbacks": callbacks,
            "configurable": {
                "user_id": meta.business_user,
                "conversation_id": conversation_id,
                "metadata": meta.model_dump(),
            },
        }

    def start(
        self,
        meta: ConversationMetadata,
        prompt_id: str,
    ) -> tuple[dict, str, RunnableWithMessageHistory]:
        """Sets up a new conversation.

        Generates a new UUID to use as the conversation ID. Gets the memory
        and LLM to use for the conversation. Stores metadata about the
        conversation. Creates the ConversationChain instance to manage the
        conversation.

        Returns the generated conversation ID, config, and ConversationChain.
        """
        conversation_id = str(uuid.uuid4())  # Generate a new UUID
        llm = get_llm(meta.provider, meta.variant)

        messages = [SYSTEM_MESSAGE]
        if prompt_id != "default":
            prompt = self.get_prompt(prompt_id)
            messages.append(HumanMessagePromptTemplate(prompt=prompt))
        else:
            messages.append(("human", "{input}"))

        chat_prompt = ChatPromptTemplate.from_messages(messages)
        chain_with_history = self.create_chain(llm, chat_prompt).with_config(
            {"metadata": {"prompt_id": prompt_id}},
        )
        self.app.state.metrics_provider.increment(Metric.COUNT_CHAT_REQUESTS, meta.model_dump())

        config = self.generate_config(meta, conversation_id, llm)
        return config, conversation_id, chain_with_history

    def get(self, conversation_id: str, user_id: str) -> tuple[dict, RunnableWithMessageHistory]:
        """
        Gets an existing conversation for the given conversation ID.

        If no metadata exists for the ID, raises a NoConversationError.

        Returns the config and created chain.
        """
        history = self.get_message_history(user_id=user_id, conversation_id=conversation_id)

        if history.metadata is not None:
            meta = history.metadata
            llm = get_llm(meta.provider, meta.variant)
            self.app.state.metrics_provider.increment(Metric.COUNT_CHAT_REQUESTS, meta.model_dump())
            config = self.generate_config(meta, conversation_id, llm)
        else:
            raise NoConversationError(conversation_id)
        prompt = ChatPromptTemplate.from_messages(
            [
                MessagesPlaceholder(variable_name="history"),
                ("human", "{input}"),
            ],
        )
        return config, self.create_chain(llm, prompt)

    def get_message_history(
        self,
        *,
        user_id: str,
        conversation_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> ChatMessageHistory:
        """
        Retrieves the chat message history for a given user and conversation.

        Args:
            user_id (str): The ID of the user.
            conversation_id (str): The ID of the conversation.
            metadata (dict[str, Any] | None, optional): Additional metadata for the conversation. Defaults to None.

        Returns:
            ChatMessageHistory: The chat message history for the user and conversation.
        """
        meta = None if metadata is None else ConversationMetadata.model_validate(metadata)
        return CosmosDBChatMessageHistory(
            self.app.state.cosmos_client,
            session_id=conversation_id,
            user_id=user_id,
            metadata=meta,
        )

    def _message_to_dict(self, message: BaseMessage) -> dict[str, str]:
        """Convert a Message to a dictionary.

        Args:
            message: Message to convert.

        Returns:
            Message as a dict.
        """
        return {"role": message.type, "content": message.content}

    def history(self, user_id: str, conversation_id: str) -> list[dict[str, str]]:
        """
        Retrieves the message history for a specific user and conversation.

        Args:
            user_id (str): The ID of the user.
            conversation_id (str): The ID of the conversation.

        Returns:
            list[dict[str, str]]: A list of dictionaries representing the message history.

        Raises:
            NoConversationError: If there is no conversation with the specified ID.
        """
        history = self.get_message_history(user_id=user_id, conversation_id=conversation_id)
        if history.messages:
            return [self._message_to_dict(m) for m in history.messages]
        raise NoConversationError(conversation_id)

    def end(self, user_id: str, conversation_id: str) -> None:
        """
        End a conversation for a specific user.

        Args:
            user_id (str): The ID of the user.
            conversation_id (str): The ID of the conversation.

        Returns:
            None

        Raises:
            NoConversationError: If no conversation is found with the given conversation ID.
        """
        history = self.get_message_history(user_id=user_id, conversation_id=conversation_id)
        if history.messages:
            return history.clear()
        raise NoConversationError(conversation_id)

    def get_prompts(self) -> dict[str, list[str]]:
        """Gets the example prompts configured for this service."""
        return self.examples

    def get_prompt(self, prompt_id: str) -> StringPromptTemplate:
        """Gets the prompt template for the given prompt ID."""
        return self.prompts[prompt_id]

    def delete_history(self, user_id: str, conversation_id: str, num_entries: int) -> None:
        """
        Delete a chosen message from the message history.

        Args:
            user_id (str): The ID of the user.
            conversation_id (str): The ID of the conversation.
            num_entries (int): The numerical index of the message to delete.

        Raises:
            NoConversationError: If no such message index exists.
        """
        history = self.get_message_history(user_id=user_id, conversation_id=conversation_id)

        if len(history.messages) > 0:
            history.delete(num_entries)
        else:
            raise NoConversationError(conversation_id)
