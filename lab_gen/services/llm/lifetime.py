from typing import Any

import boto3

from google.oauth2 import service_account
from langchain_anthropic import ChatAnthropic
from langchain_aws.chat_models.bedrock import ChatBedrock
from langchain_community.chat_models.azureml_endpoint import AzureMLChatOnlineEndpoint, CustomOpenAIChatContentFormatter
from langchain_community.llms import HuggingFaceEndpoint
from langchain_community.llms.azureml_endpoint import AzureMLEndpointApiType
from langchain_core.language_models import BaseChatModel, BaseLanguageModel
from langchain_google_vertexai import ChatVertexAI, HarmBlockThreshold, HarmCategory
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from loguru import logger

from lab_gen.datatypes.errors import ModelKeyError
from lab_gen.datatypes.models import (
    DEFAULT_MODEL_KEY,
    AzureMLModelConfig,
    AzureModelConfig,
    BedrockModelConfig,
    HuggingfaceModelConfig,
    Model,
    ModelFamily,
    ModelProvider,
    ModelVariant,
)
from lab_gen.settings import settings

MAX_TOKENS = 1536

VERTEX_SAFETY_CONFIG = {
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
}

model_providers: dict[str, Any] = {}
models: dict[str, Model] = {}


def get_llm(model_key: str = DEFAULT_MODEL_KEY) -> BaseLanguageModel:
    """
    Get the LLM for the specified model key.

    Args:
        model_key (str): The unique key for the model.

    Returns:
        BaseLanguageModel: The llm for the specified model.

    Raises:
        KeyError: If the specified model is not found.
    """
    try:
        return model_providers[model_key]
    except KeyError as ke:
        logger.warning(f"Unable to find llm model {model_key}")
        raise ModelKeyError(model_key) from ke


def get_model(model_key: str = DEFAULT_MODEL_KEY) -> Model:
    """
    Get the model definition for the specified model key.

    Args:
        model_key (str): The unique key for the model.

    Returns:
        Model: The model definition for the specified key.

    Raises:
        KeyError: If the specified model is not found.
    """
    try:
        return models[model_key]
    except KeyError as ke:
        logger.warning(f"Unable to find model definition for {model_key}")
        raise ModelKeyError(model_key) from ke


def init_azure_llm(model: Model) -> BaseChatModel:
    """
    Initialize the LLM running on Azure.

    Args:
        model (Model): The model configuration.

    Returns:
        AzureChatOpenAI: The initialized Azure LLM.
    """
    if model.family == ModelFamily.MISTRAL:
        config = AzureMLModelConfig(**model.config)
        return ChatMistralAI(
            endpoint=config.endpoint,
            api_key=config.api_key,
            max_tokens=MAX_TOKENS,
            temperature=0,
            streaming=True,
        )

    if (model.family == ModelFamily.PHI or model.family == ModelFamily.LLAMA): # noqa: PLR1714
        config = AzureMLModelConfig(**model.config)
        return AzureMLChatOnlineEndpoint(
            endpoint_url=config.endpoint,
            endpoint_api_type=AzureMLEndpointApiType.serverless,
            endpoint_api_key=config.api_key,
            content_formatter=CustomOpenAIChatContentFormatter(),
        )

    # Default to Azure OpenAI
    config = AzureModelConfig(**model.config)
    return AzureChatOpenAI(
        verbose=True,
        azure_deployment=model.identifier,
        api_version=config.api_version,
        azure_endpoint=config.endpoint,
        api_key=config.api_key,
        streaming=True,
    )


def init_bedrock_llm(model: Model) -> ChatBedrock:
    """
    Initialize the LLM running on Bedrock.

    Args:
        model (Model): The model configuration.

    Returns:
        ChatBedrock: The initialized Bedrock LLM.
    """
    config = BedrockModelConfig(**model.config)
    boto_client = boto3.client(
        service_name="bedrock-runtime",
        **config.model_dump(),
    )
    bedrock_kwargs = {
        "client": boto_client,
        "model_id": model.identifier,
        "streaming": True,
        "model_kwargs": {"max_tokens": MAX_TOKENS},
    }
    # Guardrail settings
    guardrails = {}
    guardrailid = model.config.get("guardrailIdentifier")
    guardrailversion = model.config.get("guardrailVersion")
    if guardrailid is not None and guardrailversion is not None:
        guardrails["guardrailIdentifier"] = guardrailid
        guardrails["guardrailVersion"] = guardrailversion
        bedrock_kwargs["guardrails"] = guardrails
    return ChatBedrock(**bedrock_kwargs)


def init_vertex_llm(model: Model) -> ChatVertexAI:
    """
    Initializes and returns a ChatVertexAI instance for the given model.

    Args:
        model (Model): The model configuration.

    Returns:
        ChatVertexAI: The initialized ChatVertexAI instance.

    """
    credentials = service_account.Credentials.from_service_account_info(model.config)
    vertex_setup = {
        "credentials": credentials,
        "model_name": model.identifier,
        "project": model.config["project_id"],
        "streaming": True,
        "safety_settings": VERTEX_SAFETY_CONFIG,
        "convert_system_message_to_human": True,
    }
    if "location" in model.config:
        vertex_setup["location"] = model.config["location"]
    return ChatVertexAI(**vertex_setup)


def init_github_llm(model: Model, github_token: str) -> AzureChatOpenAI:
    """
    Initializes and returns a ChatOpenAI instance for GitHub Models.

    Args:
        model (Model): The model configuration.
        github_token (str): The GitHub token for github models access.

    Returns:
        ChatOpenAI: The initialized GitHub Models LLM.
    """
    return ChatOpenAI(
        model=model.identifier,
        base_url=model.config["endpoint"],
        api_key=github_token,
        streaming=True,
    )

def init_models() -> None:
    """
    Loops through the model settings, for each model configures an LLM client.

    :param app: current fastapi application.
    """

    modelz = settings.models + settings.models_vertex
    for model in modelz:
        if model.config is not None:
            key = model.key
            llm = None
            match model.provider:
                case ModelProvider.AZURE:
                    llm = init_azure_llm(model)
                case ModelProvider.BEDROCK:
                    llm = init_bedrock_llm(model)
                case ModelProvider.VERTEX:
                    llm = init_vertex_llm(model)
                case ModelProvider.ANTHROPIC:
                    llm = ChatAnthropic(
                        model=model.identifier,
                        temperature=0,
                        max_tokens=MAX_TOKENS,
                        streaming=True,
                        max_retries=2,
                        api_key=model.config["ANTHROPIC_API_KEY"],
                    )
                case ModelProvider.HUGGINGFACE:
                    config = HuggingfaceModelConfig(**model.config)
                    llm = HuggingFaceEndpoint(
                        streaming=True,
                        repo_id=config.repo_id, huggingfacehub_api_token=config.access_token)
                case ModelProvider.GITHUB:
                    # Check for GitHub Token for use in Codespaces
                    github_token = settings.github_token

                    if github_token is not None:
                        llm = init_github_llm(model, github_token)
            if llm is not None:
                logger.debug(f"Configuring LLM for {key} {model.identifier}")
                model_providers[key] = llm
                models[key] = model

    if len(model_providers) != len(modelz):
        logger.warning(f"Configured {len(model_providers)} LLMs, but provided {len(modelz)} in settings")
