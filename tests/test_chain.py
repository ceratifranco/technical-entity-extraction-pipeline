"""Tests de la cadena con un modelo falso: no llaman a ninguna API."""

from __future__ import annotations

import sys
import traceback
from typing import Any

import pytest
from langchain_core.messages import AIMessage
from langchain_core.prompt_values import ChatPromptValue
from langchain_core.runnables import RunnableLambda
from pydantic import ValidationError

import config
from chain import (
    EntradaVaciaError,
    ExtraccionFallidaError,
    ExtractionPipeline,
    build_model,
    build_prompt,
    es_reintentable,
)
from config import ConfiguracionInvalidaError, Settings
from schemas import ExtraccionTecnica

VALIDO = ExtraccionTecnica(
    tecnologias=["FastAPI", "Redis", "PostgreSQL"],
    nivel_de_criticidad="alta",
    resumen_tecnico="API con caché en Redis y persistencia en PostgreSQL; cuello de botella.",
)


def respuesta(parsed: Any = VALIDO, finish_reason: str = "tool_calls", error: Exception | None = None):
    raw = AIMessage(content="", response_metadata={"finish_reason": finish_reason})
    return {"raw": raw, "parsed": None if error else parsed, "parsing_error": error}


class FakeModel:
    """Imita `model.with_structured_output(schema, include_raw=True)` devolviendo respuestas en orden."""

    def __init__(self, respuestas: list[Any]) -> None:
        self.respuestas = list(respuestas)
        self.llamadas = 0
        self.prompts: list[ChatPromptValue] = []

    def with_structured_output(self, schema, include_raw=False):
        assert schema is ExtraccionTecnica and include_raw

        async def responder(prompt_value: ChatPromptValue):
            self.llamadas += 1
            self.prompts.append(prompt_value)
            r = self.respuestas.pop(0)
            if isinstance(r, Exception):
                raise r
            return r

        return RunnableLambda(responder)


def pipeline(respuestas: list[Any], max_attempts: int = 3) -> tuple[ExtractionPipeline, FakeModel]:
    settings = Settings(provider="groq", groq_api_key="test", max_attempts=max_attempts)
    model = FakeModel(respuestas)
    return ExtractionPipeline(settings, model=model), model  # type: ignore[arg-type]


@pytest.fixture(autouse=True)
def sin_espera(monkeypatch):
    """Evita el backoff real entre reintentos: la espera exponencial pasa a ser cero."""
    import langchain_core.runnables.retry as retry
    from tenacity import wait_none

    monkeypatch.setattr(retry, "wait_exponential_jitter", lambda **_: wait_none())


async def test_camino_feliz():
    p, model = pipeline([respuesta()])
    assert await p.process_text("texto") == VALIDO
    assert model.llamadas == 1


async def test_prompt_incluye_texto_e_instrucciones_de_formato():
    p, model = pipeline([respuesta()])
    await p.process_text("Uso Kafka y Postgres")
    system, human = model.prompts[0].to_messages()
    assert "nivel_de_criticidad" in system.content and '"alta"' in system.content
    assert "Uso Kafka y Postgres" in human.content


def test_prompt_no_tiene_variables_sueltas():
    assert build_prompt().input_variables == ["texto"]


async def test_reintenta_si_la_respuesta_se_corta_por_tokens():
    p, model = pipeline([respuesta(finish_reason="length"), respuesta()])
    assert await p.process_text("texto") == VALIDO
    assert model.llamadas == 2


async def test_reintenta_ante_stop_reason_max_tokens_de_anthropic():
    raw = AIMessage(content="", response_metadata={"stop_reason": "max_tokens"})
    p, model = pipeline([{"raw": raw, "parsed": VALIDO, "parsing_error": None}, respuesta()])
    assert await p.process_text("texto") == VALIDO
    assert model.llamadas == 2


async def test_reintenta_si_el_json_no_valida():
    try:
        ExtraccionTecnica.model_validate({"tecnologias": []})
    except ValidationError as exc:
        error = exc
    p, model = pipeline([respuesta(error=error), respuesta()])
    assert await p.process_text("texto") == VALIDO
    assert model.llamadas == 2


async def test_reintenta_si_el_modelo_no_devuelve_estructura():
    p, model = pipeline([respuesta(parsed=None), respuesta()])
    assert await p.process_text("texto") == VALIDO
    assert model.llamadas == 2


async def test_reintenta_tool_use_failed_de_groq():
    error = RuntimeError("Error code: 400 - {'error': {'code': 'tool_use_failed'}}")
    p, model = pipeline([error, respuesta()])
    assert await p.process_text("texto") == VALIDO
    assert model.llamadas == 2


async def test_reintenta_length_finish_reason_error_de_openai():
    from openai import LengthFinishReasonError
    from openai.types.chat import ChatCompletion

    completion = ChatCompletion.model_construct(choices=[], usage=None)
    p, model = pipeline([LengthFinishReasonError(completion=completion), respuesta()])
    assert await p.process_text("texto") == VALIDO
    assert model.llamadas == 2


async def test_agota_reintentos_y_lanza_error_controlado():
    p, model = pipeline([respuesta(finish_reason="length")] * 3, max_attempts=3)
    with pytest.raises(ExtraccionFallidaError) as info:
        await p.process_text("texto")
    assert info.value.intentos == 3
    assert model.llamadas == 3


async def test_no_reintenta_errores_definitivos():
    p, model = pipeline([PermissionError("401 invalid api key"), respuesta()])
    with pytest.raises(ExtraccionFallidaError) as info:
        await p.process_text("texto")
    assert info.value.intentos == 1
    assert model.llamadas == 1


async def test_texto_vacio_no_llama_al_modelo():
    p, model = pipeline([respuesta()])
    with pytest.raises(EntradaVaciaError):
        await p.process_text("   ")
    assert model.llamadas == 0


def test_clasificacion_de_errores():
    assert es_reintentable(RuntimeError("json_validate_failed"))
    assert not es_reintentable(RuntimeError("invalid_api_key"))


def test_settings_exige_key_del_proveedor():
    with pytest.raises(ValidationError):
        Settings(provider="anthropic", groq_api_key="x")


@pytest.fixture
def entorno_limpio(monkeypatch):
    """Ignora el .env real y las variables del entorno de quien corre los tests."""
    monkeypatch.setattr(config, "load_dotenv", lambda: None)
    for var in config.ENV_VARS.values():
        monkeypatch.delenv(var, raising=False)
    return monkeypatch


def test_from_env_sin_key_da_error_de_configuracion(entorno_limpio):
    with pytest.raises(ConfiguracionInvalidaError, match="Falta la API key.*ANTHROPIC_API_KEY"):
        Settings.from_env(provider="anthropic")


def test_from_env_nombra_la_variable_invalida(entorno_limpio):
    entorno_limpio.setenv("GROQ_API_KEY", "x")
    entorno_limpio.setenv("LLM_MAX_TOKENS", "abc")
    with pytest.raises(ConfiguracionInvalidaError, match="LLM_MAX_TOKENS"):
        Settings.from_env()


def test_build_model_sin_paquete_del_proveedor(monkeypatch):
    monkeypatch.setitem(sys.modules, "langchain_anthropic", None)
    settings = Settings(provider="anthropic", anthropic_api_key="x")
    with pytest.raises(ConfiguracionInvalidaError, match="pip install langchain-anthropic"):
        build_model(settings)


def test_error_al_crear_el_modelo_no_expone_la_api_key():
    pytest.importorskip("langchain_groq")
    # model_construct saltea la validación de Settings: el error lo lanza el ChatGroq real.
    valores = Settings(provider="groq", groq_api_key="gsk_secreto").model_dump()
    settings = Settings.model_construct(**{**valores, "max_tokens": "abc"})
    with pytest.raises(ConfiguracionInvalidaError) as info:
        build_model(settings)
    assert "max_tokens" in str(info.value)
    assert "gsk_secreto" not in "".join(traceback.format_exception(info.value))


def test_error_de_configuracion_no_expone_keys_en_el_traceback(entorno_limpio):
    entorno_limpio.setenv("GROQ_API_KEY", "gsk_secreto")
    with pytest.raises(ConfiguracionInvalidaError) as info:
        Settings.from_env(provider="anthropic")
    assert isinstance(info.value.__cause__, ValidationError)
    # pydantic recorta input_value, así que de una key puede quedar sólo el principio: "gsk_sec...".
    assert "gsk_" not in "".join(traceback.format_exception(info.value))


def test_build_model_import_roto_conserva_la_causa(monkeypatch):
    import builtins

    importar = builtins.__import__

    def falla(name, *args, **kwargs):
        if name == "langchain_groq":
            raise ImportError("cannot import name 'X' from 'langchain_core'", name="langchain_core")
        return importar(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", falla)
    settings = Settings(provider="groq", groq_api_key="x")
    with pytest.raises(
        ConfiguracionInvalidaError, match="No se pudo cargar el proveedor 'groq': cannot import"
    ) as info:
        build_model(settings)
    assert isinstance(info.value.__cause__, ImportError)


def test_build_model_sin_sdk_muestra_el_error_original(monkeypatch):
    langchain_groq = pytest.importorskip("langchain_groq")

    def falla(**kwargs):
        raise ImportError("Could not import groq python package. Please install it with `pip install groq`.")

    monkeypatch.setattr(langchain_groq, "ChatGroq", falla)
    settings = Settings(provider="groq", groq_api_key="x")
    with pytest.raises(ConfiguracionInvalidaError, match="pip install groq"):
        build_model(settings)
