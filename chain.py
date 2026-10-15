"""Pipeline LCEL: prompt | model.with_structured_output(ExtraccionTecnica) | validación, con reintentos.

    texto ──► ChatPromptTemplate ──► LLM con salida estructurada ──► _validar_salida ──► ExtraccionTecnica
                    ▲                                                     │
                    └──────────── .with_retry() si la salida es inválida ◄┘
"""

from __future__ import annotations

import logging
from contextvars import ContextVar
from typing import Any

from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import ValidationError

from config import Settings
from schemas import ExtraccionTecnica

logger = logging.getLogger("pipeline")

# finish_reason (OpenAI/Groq) o stop_reason (Anthropic) que indican respuesta cortada.
FINISH_REASONS_TRUNCADOS = {"length", "max_tokens"}

# Códigos con los que Groq rechaza una tool call con JSON inválido o que no respeta el schema.
CODIGOS_JSON_INVALIDO = ("tool_use_failed", "json_validate_failed")


# --------------------------------------------------------------------------- errores


class SalidaInvalidaError(Exception):
    """El LLM respondió, pero la salida no sirve: se puede reintentar."""


class RespuestaIncompletaError(SalidaInvalidaError):
    """El modelo cortó la respuesta por límite de tokens (finish_reason = length)."""


class SalidaMalFormadaError(SalidaInvalidaError):
    """JSON mal formado, campos faltantes o valores que no pasan la validación Pydantic."""


class ExtraccionFallidaError(Exception):
    """Se agotaron los reintentos o hubo un error no reintentable."""

    def __init__(self, message: str, intentos: int) -> None:
        super().__init__(message)
        self.intentos = intentos


def es_reintentable(exc: BaseException) -> bool:
    """Decide qué errores dispara `.with_retry()`: sólo los de salida inválida.

    Una API key inválida o un modelo inexistente fallan igual en cada intento, así
    que no se reintentan. Los errores HTTP transitorios (429, 5xx) ya los reintenta
    el SDK del proveedor (`max_retries`).
    """
    if isinstance(exc, (SalidaInvalidaError, OutputParserException, ValidationError)):
        return True
    # Con method="json_schema", el SDK de OpenAI lanza esto si finish_reason == "length",
    # antes de que la respuesta llegue a _validar_salida.
    if type(exc).__name__ == "LengthFinishReasonError":
        return True
    # Groq valida la tool call del lado del servidor y devuelve 400 si el JSON está roto.
    return any(codigo in str(exc) for codigo in CODIGOS_JSON_INVALIDO)


def _debe_reintentar(exc: BaseException) -> bool:
    """Predicado de `.with_retry()`: igual que `es_reintentable`, pero deja registro del motivo."""
    reintentable = es_reintentable(exc)
    if reintentable:
        logger.warning("Intento fallido (%s): %s", type(exc).__name__, _resumir(exc))
    return reintentable


# --------------------------------------------------------------------------- modelo


def build_model(settings: Settings) -> BaseChatModel:
    """Crea el chat model de LangChain según `LLM_PROVIDER`, como el AsyncLLMManager del Módulo 1."""
    api_key, model = settings.credentials()
    common: dict[str, Any] = {
        "model": model,
        "api_key": api_key,
        "max_tokens": settings.max_tokens,
        "timeout": settings.timeout_s,
        "max_retries": settings.max_retries,
    }
    match settings.provider:
        case "groq":
            from langchain_groq import ChatGroq

            return ChatGroq(**common, temperature=settings.temperature)
        case "openai":
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(**common, temperature=settings.temperature)
        case "anthropic":
            from langchain_anthropic import ChatAnthropic

            # Igual que en el Módulo 1: no se envía temperature a Anthropic.
            return ChatAnthropic(**common)


# --------------------------------------------------------------------------- prompt


SYSTEM_PROMPT = """\
Sos un ingeniero de software senior que analiza textos técnicos: descripciones de \
arquitectura, logs de error, reportes de incidentes o tickets.

Tu tarea es extraer entidades técnicas del texto del usuario.

Reglas:
- Usá sólo información presente en el texto. No inventes tecnologías.
- Si una tecnología se infiere sin ambigüedad (ej. "psycopg2" implica PostgreSQL), incluila.
- Si el texto es ambiguo, elegí la criticidad más razonable y aclaralo en el resumen.
- Respondé siempre en español.

Formato de salida:
{format_instructions}"""

HUMAN_PROMPT = """\
Texto a analizar:
<texto>
{texto}
</texto>"""


def instrucciones_de_formato(schema: type[ExtraccionTecnica] = ExtraccionTecnica) -> str:
    """Genera las instrucciones de formato a partir del schema Pydantic (una sola fuente de verdad)."""
    json_schema = schema.model_json_schema()
    lineas = []
    for nombre, campo in json_schema["properties"].items():
        if "$ref" in campo:  # enum
            ref = json_schema["$defs"][campo["$ref"].split("/")[-1]]
            tipo = " | ".join(f'"{v}"' for v in ref["enum"])
        elif campo.get("type") == "array":
            tipo = "lista de strings (al menos 1)"
        else:
            tipo = campo.get("type", "string")
        lineas.append(f"- {nombre} ({tipo}): {campo.get('description', '')}")
    return "Devolvé un objeto con exactamente estos campos:\n" + "\n".join(lineas)


def build_prompt() -> ChatPromptTemplate:
    """Template modular: recibe `texto`; las instrucciones de formato quedan precargadas."""
    prompt = ChatPromptTemplate.from_messages([("system", SYSTEM_PROMPT), ("human", HUMAN_PROMPT)])
    return prompt.partial(format_instructions=instrucciones_de_formato())


# --------------------------------------------------------------------------- validación

# Contador de intentos por llamada a process_text(). Es un ContextVar para que cada
# llamada concurrente lleve su propia cuenta.
_intentos: ContextVar[list[int] | None] = ContextVar("intentos", default=None)


def _registrar_intento(prompt_value: Any) -> Any:
    contador = _intentos.get()
    if contador is not None:
        contador[0] += 1
        if contador[0] > 1:
            logger.info("Reintento #%d: se vuelve a llamar al modelo", contador[0] - 1)
    return prompt_value


def _finish_reason(raw: AIMessage | None) -> str | None:
    if raw is None:
        return None
    metadata = raw.response_metadata or {}
    return metadata.get("finish_reason") or metadata.get("stop_reason")


def _validar_salida(resultado: dict[str, Any]) -> ExtraccionTecnica:
    """Recibe {raw, parsed, parsing_error} de `with_structured_output(include_raw=True)`.

    Primero mira el finish_reason, antes de confiar en el objeto: si la respuesta se
    cortó por tokens, el JSON puede estar incompleto aunque el parser no se queje.
    """
    raw: AIMessage | None = resultado.get("raw")
    finish_reason = _finish_reason(raw)
    usage = getattr(raw, "usage_metadata", None) or {}
    logger.info(
        "Respuesta del modelo: finish_reason=%s, tokens=%s→%s",
        finish_reason,
        usage.get("input_tokens", "?"),
        usage.get("output_tokens", "?"),
    )

    if finish_reason in FINISH_REASONS_TRUNCADOS:
        raise RespuestaIncompletaError(
            f"El modelo cortó la respuesta por límite de tokens (finish_reason={finish_reason})"
        )

    if (error := resultado.get("parsing_error")) is not None:
        raise SalidaMalFormadaError(f"La salida no cumple el schema: {_resumir(error)}") from error

    parsed = resultado.get("parsed")
    if parsed is None:
        raise SalidaMalFormadaError("El modelo respondió en texto libre, sin la estructura pedida")

    # Revalidación explícita: si el parser devolvió un dict (p.ej. json_mode), lo convierte.
    resultado_validado = ExtraccionTecnica.model_validate(parsed, from_attributes=True)
    logger.info("Validación OK: %d tecnologías", len(resultado_validado.tecnologias))
    return resultado_validado


def _resumir(error: BaseException) -> str:
    texto = " ".join(str(error).split())
    return texto if len(texto) <= 200 else texto[:200] + "…"


# --------------------------------------------------------------------------- cadena


def build_chain(model: BaseChatModel, max_attempts: int = 3) -> Runnable[dict, ExtraccionTecnica]:
    """Ensambla prompt | model.with_structured_output(Schema) | validación, con `.with_retry()`."""
    chain = (
        build_prompt()
        | RunnableLambda(_registrar_intento, name="registrar_intento")
        | model.with_structured_output(ExtraccionTecnica, include_raw=True)
        | RunnableLambda(_validar_salida, name="validar_salida")
    )
    return chain.with_retry(
        retry_if_exception_type=_debe_reintentar,
        stop_after_attempt=max_attempts,
        wait_exponential_jitter=True,
        exponential_jitter_params={"max": 4.0},
    )


class ExtractionPipeline:
    """Arma el modelo y la cadena una vez y los reutiliza en cada llamada."""

    def __init__(self, settings: Settings, model: BaseChatModel | None = None) -> None:
        self.settings = settings
        self.model = model or build_model(settings)
        self.chain = build_chain(self.model, settings.max_attempts)

    async def process_text(self, text: str) -> ExtraccionTecnica:
        """Extrae entidades técnicas de `text` y devuelve el objeto ya validado.

        Lanza `ExtraccionFallidaError` si se agotan los reintentos o el error no es reintentable.
        """
        if not text or not text.strip():
            raise ValueError("El texto de entrada está vacío")

        contador = [0]
        token = _intentos.set(contador)
        logger.info(
            "Procesando texto (%d caracteres) con %s/%s",
            len(text),
            self.settings.provider,
            self.settings.credentials()[1],
        )
        try:
            resultado = await self.chain.ainvoke({"texto": text})
        except Exception as exc:
            reintentable = es_reintentable(exc)
            logger.error(
                "Extracción fallida tras %d intento(s) (%s): %s",
                contador[0],
                "reintentos agotados" if reintentable else "error no reintentable",
                _resumir(exc),
            )
            raise ExtraccionFallidaError(_resumir(exc), contador[0]) from exc
        finally:
            _intentos.reset(token)

        logger.info("Extracción OK en %d intento(s)", contador[0])
        return resultado


_pipeline: ExtractionPipeline | None = None


async def process_text(text: str) -> ExtraccionTecnica:
    """Atajo: usa un pipeline configurado desde `.env` (se crea en la primera llamada)."""
    global _pipeline
    if _pipeline is None:
        _pipeline = ExtractionPipeline(Settings.from_env())
    return await _pipeline.process_text(text)
