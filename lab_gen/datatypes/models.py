from enum import Enum

from pydantic import BaseModel, Field, computed_field


class ModelVariant(Enum):
    """Represents different model variants."""

    GENERAL = "GENERAL"
    ADVANCED = "ADVANCED"
    MULTIMODAL = "MULTIMODAL"
    EXPERIMENTAL = "EXPERIMENTAL"


class ModelProvider(Enum):
    """Represents different model providers."""

    AZURE = "AZURE"
    ANTHROPIC = "ANTHROPIC"
    BEDROCK = "BEDROCK"
    VERTEX = "VERTEX"
    HUGGINGFACE = "HUGGINGFACE"
    GITHUB = "GITHUB"


class ModelFamily(Enum):
    """Represents the different model families."""
    GPT = "GPT"
    CLAUDE = "CLAUDE"
    GEMINI = "GEMINI"
    MISTRAL = "MISTRAL"
    LLAMA = "LLAMA"
    PHI = "PHI"
    UNSPECIFIED = "UNSPECIFIED"


class Model(BaseModel):
    """
    Represents a model definition.

    Attributes:
        provider (ModelProvider): The model providers.
        variant (ModelVariant): The variants of the model.
        family (ModelFamily): The family of the model.
        description (str): The description of the model.
        location (str):  The geographic location where the model is running.
    """

    provider: ModelProvider
    variant: ModelVariant
    family: ModelFamily = Field(default=ModelFamily.UNSPECIFIED)
    description: str | None = None
    location: str
    identifier: str = Field(exclude=True)
    config: dict[str, str] = Field(exclude=True)

    @staticmethod
    def compute_key(provider: ModelProvider, variant: ModelVariant, family: ModelFamily) -> str:
        """
        Compute the key for the model.

        Args:
            provider (ModelProvider): The model provider.
            variant (ModelVariant): The model variant.
            family (ModelFamily): The model family.

        Returns:
            str: The computed key.
        """
        return provider.value + family.value + variant.value

    @computed_field
    def key(self) -> str:
        """
        The unique key for the model.

        Returns:
            str: The computed key.
        """
        return Model.compute_key(self.provider, self.variant, self.family)


class AzureModelConfig(BaseModel):
    """Represents an Azure model config."""

    endpoint: str = Field(alias="AZURE_OPENAI_ENDPOINT")
    api_key: str = Field(alias="AZURE_OPENAI_API_KEY")
    api_version: str = Field(alias="AZURE_OPENAI_API_VERSION")

class AzureMLModelConfig(BaseModel):
    """Represents an Azure ML Serverless model config."""

    endpoint: str = Field(alias="AZUREAI_ENDPOINT")
    api_key: str = Field(alias="AZUREAI_API_KEY")

class BedrockModelConfig(BaseModel):
    """Represents a model config for AWS Bedrock runtime."""

    region_name: str = Field(alias="AWS_REGION")
    aws_access_key_id: str = Field(alias="AWS_ACCESS_KEY_ID")
    aws_secret_access_key: str = Field(alias="AWS_SECRET_ACCESS_KEY")


class HuggingfaceModelConfig(BaseModel):
    """Represents a model config for Huggingface."""
    repo_id: str = Field(alias="HF_REPO_ID")
    access_token: str = Field(alias="HF_TOKEN")


DEFAULT_MODEL_KEY = Model.compute_key(ModelProvider.AZURE, ModelVariant.GENERAL, ModelFamily.GPT)
