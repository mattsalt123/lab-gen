import enum
import os

from pathlib import Path
from tempfile import gettempdir
from typing import Any

from azure.appconfiguration import AzureAppConfigurationClient
from azure.core.exceptions import ResourceNotFoundError
from loguru import logger
from pydantic import Field, TypeAdapter
from pydantic.fields import FieldInfo
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict

from lab_gen.datatypes.models import Model


TEMP_DIR = Path(gettempdir())
APP_DIR = Path(__file__).resolve().parent


class SettingsError(Exception):
    """SettingsError."""


class LogLevel(str, enum.Enum):
    """Possible log levels."""

    NOTSET = "NOTSET"
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    FATAL = "FATAL"


class AzureSettingsSource(PydanticBaseSettingsSource):
    """A settings source that gets it data from Azure App Configuration."""

    def __init__(self, settings_cls: type[BaseSettings]) -> None:
        super().__init__(settings_cls)

        # The environment variable that contains the connection string
        self._azure_appconfig = os.getenv("APPCONFIGURATION_CONNECTION_STRING")
        # The field.alais needs to start with this prefix so it can be looked up
        self._azure_prefix = self.config.get("azure_prefix", "AZURE")
        self.environment = os.getenv(f"{self.config['env_prefix']}ENVIRONMENT")

    def __call__(self) -> dict[str, Any]:  # noqa: D102
        data: dict[str, Any] = {}

        if self._azure_appconfig is None:
            return data

        self.client = AzureAppConfigurationClient.from_connection_string(
            self._azure_appconfig,
        )

        for field_name, field in self.settings_cls.model_fields.items():
            try:
                if (field.alias is not None) and (field.alias.startswith(self._azure_prefix)):
                    field_value, field_key, value_is_complex = self.get_field_value(
                        field,
                        field_name,
                    )
                    if field_value is not None:
                        field.default = field_value
                        data[field_key] = field_value
            except Exception as e:
                msg = f'error getting value for field "{field_name}" from source "{self.__class__.__name__}"'
                logger.exception(msg)
                raise SettingsError(
                    msg,
                ) from e

        return data

    def get_field_value(  # noqa: D102
        self,
        field: FieldInfo,
        field_name: str,  # noqa: ARG002
    ) -> tuple[Any, str, bool]:
        try:
            field_value = (
                self.client.get_configuration_setting(key=field.alias, label=self.environment)
                if field.alias is not None
                else None
            )
        except ResourceNotFoundError as rnfe:
            logger.warning(f"Field {field.alias} not found in azure appconfig: {rnfe.message}")
            field_value = None

        if (field_value is not None) and (field_value.value is not None):
            if field_value.content_type == "application/json":
                ta = TypeAdapter(list[Model])
                model_list = ta.validate_json(field_value.value)
                field_value = model_list
            else:
                field_value = field_value.value
        return field_value, field.alias, False


class Settings(BaseSettings):
    """
    Application settings.

    These parameters can be configured
    with environment variables.
    """

    host: str = "127.0.0.1"
    port: int = 8080
    # quantity of workers for uvicorn
    workers_count: int = 1
    # Enable uvicorn reloading
    reload: bool = False

    # Add version
    version: str = "0.0.0"

    # Current environment
    environment: str = "dev"

    log_level: LogLevel = Field(alias="LOGURU_LEVEL", default=LogLevel.INFO)

    static_dir: Path = APP_DIR / "static"
    prompts_dir: Path = APP_DIR / "templates"
    chat_history_dir: Path = APP_DIR.parent / "filestorage"
    # Endpoint for opentelemetry.
    # E.G. http://localhost:4317
    opentelemetry_endpoint: str | None = None

    rate_limit_default: str = Field("100/minute", alias="RATE_LIMIT_DEFAULT")

    api_key: str = Field(alias="AZURE_APP_API_KEY")
    azure_monitor_connection_string: str | None = Field(default=None, alias="APPLICATIONINSIGHTS_CONNECTION_STRING")
    models: list[Model] = Field(alias="AZURE_MODELS")
    models_vertex: list[Model] = Field(default=[], alias="AZURE_MODELS_VERTEX")
    github_token: str | None = Field(default=None, alias="GITHUB_TOKEN")

    session_store_uri: str
    session_store_key: str
    # Defaults to 3 days (3 * 24 * 60 * 60)
    session_store_ttl: int = 259200

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="LAB_GEN_",
        env_file_encoding="utf-8",
        secrets_dir=APP_DIR.parent / "secrets",
        extra="allow",
    )

    @classmethod
    def settings_customise_sources(
        cls: type[BaseSettings],
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """
        Customizes the order of settings sources for the given `settings_cls`.

        Args:
            cls (type[BaseSettings]): The class of the settings.
            settings_cls (type[BaseSettings]): The class of the settings.
            init_settings (PydanticBaseSettingsSource): The initial settings source.
            env_settings (PydanticBaseSettingsSource): The environment settings source.
            dotenv_settings (PydanticBaseSettingsSource): The dotenv settings source.
            file_secret_settings (PydanticBaseSettingsSource): The file secret settings source.

        Returns:
            tuple[PydanticBaseSettingsSource, ...]: The customized order of settings sources.
        """
        # Here we choose the order for settings
        return (
            init_settings,
            file_secret_settings,
            env_settings,
            dotenv_settings,
            AzureSettingsSource(settings_cls),
        )


settings = Settings()
