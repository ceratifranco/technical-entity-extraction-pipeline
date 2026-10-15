"""Configuración desde `.env`: mismo patrón `Settings` que el cliente del Módulo 1."""

from __future__ import annotations

import os
from typing import Literal, Self

from dotenv import load_dotenv
from pydantic import BaseModel, Field, SecretStr, model_validator

ProviderName = Literal["groq", "openai", "anthropic"]

# Campo de Settings → variable de entorno
ENV_VARS = {
    "provider": "LLM_PROVIDER",
    "groq_api_key": "GROQ_API_KEY",
    "openai_api_key": "OPENAI_API_KEY",
    "anthropic_api_key": "ANTHROPIC_API_KEY",
    "groq_model": "GROQ_MODEL",
    "openai_model": "OPENAI_MODEL",
    "anthropic_model": "ANTHROPIC_MODEL",
    "temperature": "LLM_TEMPERATURE",
    "max_tokens": "LLM_MAX_TOKENS",
    "timeout_s": "LLM_TIMEOUT_S",
    "max_retries": "LLM_MAX_RETRIES",
    "max_attempts": "PIPELINE_MAX_ATTEMPTS",
}


class Settings(BaseModel):
    provider: ProviderName = "groq"

    groq_api_key: SecretStr | None = None
    openai_api_key: SecretStr | None = None
    anthropic_api_key: SecretStr | None = None

    groq_model: str = "openai/gpt-oss-20b"
    openai_model: str = "gpt-4o-mini"
    anthropic_model: str = "claude-haiku-4-5"

    temperature: float = Field(0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(2048, gt=0, le=32_000)
    timeout_s: float = Field(30.0, gt=0)
    # Reintentos del SDK ante errores HTTP transitorios (429, 5xx, red).
    max_retries: int = Field(2, ge=0, le=10)
    # Intentos de la cadena completa ante salida mal formada o incompleta (.with_retry).
    max_attempts: int = Field(3, ge=1, le=10)

    @model_validator(mode="after")
    def _key_for_selected_provider(self) -> Self:
        key = getattr(self, f"{self.provider}_api_key")
        if key is None or not key.get_secret_value().strip():
            raise ValueError(
                f"Falta la API key para el proveedor '{self.provider}' "
                f"(variable {self.provider.upper()}_API_KEY)"
            )
        return self

    @classmethod
    def from_env(cls, **overrides: object) -> Self:
        """Carga `.env` y arma la configuración. `overrides` pisa lo que venga del entorno."""
        load_dotenv()
        data: dict[str, object] = {
            field: os.environ[var] for field, var in ENV_VARS.items() if os.environ.get(var)
        }
        data.update({k: v for k, v in overrides.items() if v is not None})
        return cls.model_validate(data)

    def credentials(self) -> tuple[str, str]:
        """Devuelve (api_key, modelo) del proveedor seleccionado."""
        key: SecretStr = getattr(self, f"{self.provider}_api_key")
        return key.get_secret_value(), getattr(self, f"{self.provider}_model")
