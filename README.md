# Pipeline de Extracción de Entidades Técnicas

Pre-entrega 2. Recibe un párrafo de texto sin procesar, como una descripción de arquitectura o un log de error, y devuelve un objeto **validado con Pydantic** con las tecnologías mencionadas, el nivel de criticidad y un resumen técnico.

```text
texto ─► ChatPromptTemplate ─► LLM.with_structured_output(ExtraccionTecnica) ─► validación ─► objeto validado
               ▲                                                                       │
               └───────────────── .with_retry() si la salida es inválida ◄─────────────┘
```

> **Se puede probar gratis con Groq**, que no pide tarjeta. OpenAI y Anthropic son opcionales.

## Componentes

| Requisito | Dónde |
|---|---|
| Esquema Pydantic: `tecnologias`, `nivel_de_criticidad` (enum), `resumen_tecnico` | `schemas.py` → `ExtraccionTecnica` |
| Prompt template modular (texto + instrucciones de formato) | `chain.py` → `build_prompt()` |
| Cadena LCEL: Prompt + LLM + parser | `chain.py` → `build_chain()` |
| Reintento automático ante JSON mal formado o incompleto | `chain.py` → `.with_retry()` + `es_reintentable()` |
| Función asíncrona `process_text(text)` con `.ainvoke()` | `chain.py` → `process_text()` |
| Logs de validación y reintentos | logger `pipeline` |
| Cliente ChatOpenAI / ChatAnthropic, reutilizando la lógica del Módulo 1 | `config.py` (`Settings`) + `chain.py` → `build_model()` |
| Mini-script de prueba asíncrono | `main.py` |

### `schemas.py`: el contrato

```python
class NivelCriticidad(StrEnum):
    BAJA = "baja"; MEDIA = "media"; ALTA = "alta"

class ExtraccionTecnica(BaseModel):
    tecnologias: list[str]              # 1 a 30 elementos, sin vacíos ni duplicados
    nivel_de_criticidad: NivelCriticidad
    resumen_tecnico: str                # 20 a 600 caracteres
```

Restricciones del contrato:

- `tecnologias` **no puede estar vacía**. Se le quitan espacios, strings vacíos y duplicados sin distinguir mayúsculas (`["Redis", " redis "]` → `["Redis"]`).
- `nivel_de_criticidad` acepta `"Alta"` o `" alta "` y lo normaliza, pero rechaza valores fuera del enum, como `"crítica"`.
- `resumen_tecnico` tiene longitud acotada, para descartar respuestas vacías o desbordadas.
- `extra="forbid"`: un campo inventado por el modelo es un error.
- Las `description` de cada campo viajan al LLM dentro del schema de la tool call, y además generan las instrucciones de formato del prompt. El schema es la única fuente de verdad.

### `chain.py`: la cadena

```python
chain = (
    build_prompt()                                                    # ChatPromptTemplate
    | RunnableLambda(_registrar_intento)                              # log del número de intento
    | model.with_structured_output(ExtraccionTecnica, include_raw=True)
    | RunnableLambda(_validar_salida)                                 # finish_reason + parsing + Pydantic
).with_retry(
    retry_if_exception_type=_debe_reintentar,
    stop_after_attempt=settings.max_attempts,                        # PIPELINE_MAX_ATTEMPTS, 3 por defecto
    wait_exponential_jitter=True,                                     # backoff exponencial con jitter, tope 4 s
    exponential_jitter_params={"max": 4.0},
)
```

`retry_if_exception_type` recibe un predicado, cosa que `with_retry` acepta recién desde langchain-core 1.6.9: por eso ese es el piso en `requirements.txt`.

La base es `prompt | model.with_structured_output(Schema)`. Se le agregan dos cosas:

1. **`include_raw=True`**. Sin esto, LangChain devuelve sólo el objeto parseado y se pierde el `AIMessage` original, que trae el **`finish_reason`**. Con `include_raw=True`, un error de parseo no corta la cadena: queda guardado en `parsing_error`. Así `_validar_salida` decide con toda la información y revisa en orden:
   1. Si `finish_reason` es `length` (OpenAI/Groq) o `stop_reason` es `max_tokens` (Anthropic), la respuesta se cortó por falta de tokens. Se lanza `RespuestaIncompletaError` **antes** de aceptar el objeto, aunque el parser haya logrado armar algo con lo que llegó.
   2. Si hubo `parsing_error` (JSON roto o que no valida contra Pydantic), se lanza `SalidaMalFormadaError`.
   3. Si `parsed` es `None` (el modelo respondió en texto libre, sin tool call), también se lanza `SalidaMalFormadaError`.
2. **`.with_retry()` con un predicado**. Sólo se reintentan los errores de salida inválida. Una API key inválida o un modelo inexistente fallan igual en todos los intentos, así que se cortan en el primero. Los errores HTTP transitorios (429, 5xx) los reintenta el SDK del proveedor con `max_retries`.

| Error | ¿Reintenta la cadena? |
|---|---|
| Respuesta cortada por tokens (`finish_reason=length` / `max_tokens`) | Sí |
| OpenAI lanza `LengthFinishReasonError` (respuesta cortada en modo `json_schema`) | Sí |
| JSON mal formado o que no valida contra el schema | Sí |
| El modelo no devolvió la estructura (sin tool call) | Sí |
| Groq rechaza la tool call (`400 tool_use_failed` / `json_validate_failed`) | Sí |
| API key inválida, modelo inexistente, request inválido | No: falla en el primer intento |

Cuando se agotan los intentos, `process_text()` lanza `ExtraccionFallidaError`. Ese error trae el motivo y la cantidad de intentos.

Otros dos errores se lanzan antes de llamar al modelo:

| Error | Cuándo |
|---|---|
| `EntradaVaciaError` (subclase de `ValueError`) | El texto está vacío o sólo tiene espacios |
| `ConfiguracionInvalidaError` (de `config.py`, subclase de `ValueError`) | `Settings.from_env()` o `build_model()`: falta la API key, un valor del entorno no es válido o falta el paquete del proveedor. Si el error viene de `Settings`, el mensaje nombra la variable (p.ej. `LLM_MAX_TOKENS`). Ni el mensaje ni el traceback incluyen la API key |

Si se construye `Settings(...)` a mano, en lugar de `Settings.from_env()`, los errores de validación llegan como `pydantic.ValidationError` (también sin las API keys: `Settings` usa `hide_input_in_errors=True`).

El prompt es un `ChatPromptTemplate` sin f-strings. Recibe `{texto}`, y `{format_instructions}` queda precargado con `.partial()` a partir del schema.

### `config.py`: reutiliza el Módulo 1

Usa el mismo patrón `Settings` del [cliente async del Módulo 1](https://github.com/ceratifranco/unified-async-llm-client): un modelo Pydantic que se carga desde `.env`, elige el proveedor con `LLM_PROVIDER` y **falla al arrancar** si falta la API key del proveedor elegido. `build_model()` crea `ChatGroq`, `ChatOpenAI` o `ChatAnthropic` según esa configuración, igual que lo hacía `AsyncLLMManager`.

## Instalación

Requiere Python 3.12.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env      # completá al menos GROQ_API_KEY
```

Con [uv](https://docs.astral.sh/uv/):

```bash
uv venv -p 3.12 && source .venv/bin/activate && uv pip install -r requirements.txt
```

| Variable | Default | Descripción |
|---|---|---|
| `LLM_PROVIDER` | `groq` | `groq`, `openai` o `anthropic` |
| `GROQ_API_KEY` / `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` | — | La del proveedor elegido es obligatoria. La de Groq es gratis: https://console.groq.com/keys |
| `GROQ_MODEL` / `OPENAI_MODEL` / `ANTHROPIC_MODEL` | `openai/gpt-oss-20b` / `gpt-4o-mini` / `claude-haiku-4-5` | |
| `LLM_TEMPERATURE` | `0` | |
| `LLM_MAX_TOKENS` | `2048` | |
| `LLM_TIMEOUT_S` | `30` | |
| `LLM_MAX_RETRIES` | `2` | Reintentos del SDK ante 429, 5xx y errores de red |
| `PIPELINE_MAX_ATTEMPTS` | `3` | Intentos de la cadena (`.with_retry`) ante salida inválida |

## Uso

### Mini-script de prueba

```bash
python main.py                    # 3 ejemplos: arquitectura, log de error y texto ambiguo
python main.py --texto "..."      # un texto propio
python main.py --concurrente      # los 3 ejemplos en paralelo con asyncio.gather
python main.py --demo-truncado    # max_tokens=40: fuerza la respuesta cortada y los reintentos
python main.py --provider openai  # otro proveedor (necesita su key)
python main.py -q                 # sin logs
```

### Como librería

```python
import asyncio
from chain import process_text

resultado = asyncio.run(process_text("La API en FastAPI usa Redis y PostgreSQL..."))
print(resultado.model_dump_json(indent=2))
```

Para usar otra configuración, creá el pipeline a mano: `ExtractionPipeline(Settings.from_env(provider="anthropic"))`.

## Ejemplo de salida

**Entrada:** log de error

```text
2026-10-14 03:12:55 ERROR [payments-worker] sqlalchemy.exc.OperationalError: (psycopg2.OperationalError)
connection to server at "db-prod.internal" (10.0.3.12), port 5432 failed: FATAL: remaining connection
slots are reserved for non-replication superuser connections
2026-10-14 03:12:55 ERROR [payments-worker] Task charge_customer[8f2c] raised unexpected: OperationalError
— retry 5/5 exhausted, message moved to DLQ (kafka topic payments.dlq)
```

**Salida** (Groq, `openai/gpt-oss-20b`):

```json
{
  "tecnologias": ["SQLAlchemy", "psycopg2", "PostgreSQL", "Kafka"],
  "nivel_de_criticidad": "alta",
  "resumen_tecnico": "El worker de pagos falló al intentar conectarse a la base de datos PostgreSQL, ya que los slots de conexión se agotaron. El error de SQLAlchemy/psycopg2 provocó que la tarea charge_customer se marcara como fallida y el mensaje se enviara al DLQ de Kafka. Se requiere revisar la configuración de conexiones y la capacidad de la base de datos."
}
```

**Entrada:** descripción de arquitectura

```json
{
  "tecnologias": ["FastAPI", "Kubernetes", "AWS EKS", "Redis", "PostgreSQL", "Alembic", "Celery", "RabbitMQ"],
  "nivel_de_criticidad": "media",
  "resumen_tecnico": "El backend, una API REST en FastAPI desplegada en Kubernetes sobre AWS EKS, utiliza Redis como caché de sesiones y PostgreSQL como base de datos principal. Durante picos de tráfico, Redis se queda sin memoria, provocando que todas las lecturas se dirijan a PostgreSQL y elevando la latencia p99 a 4 s."
}
```

### Logs

```text
20:51:09 INFO    pipeline: Procesando texto (376 caracteres) con groq/openai/gpt-oss-20b
20:51:10 INFO    pipeline: Respuesta del modelo: finish_reason=tool_calls, tokens=712→346
20:51:10 INFO    pipeline: Validación OK: 8 tecnologías
20:51:10 INFO    pipeline: Extracción OK en 1 intento(s)
```

## Prueba de estrés

**1. Texto ambiguo.** *"Ayer algo andaba lento, creo que era la base o quizás el cache, no sé bien. Alguien reinició el contenedor…"*

El validador no lanza excepciones y el modelo devuelve un objeto válido en el primer intento: criticidad `media` y un resumen que aclara la incertidumbre. Pero como el contrato exige al menos una tecnología, el modelo completa la lista con lo más cercano que encuentra: `["contenedor"]` en una corrida y `["Docker"]` en otra, aunque el texto nunca nombra Docker. **Conclusión:** la validación garantiza la *forma* del objeto, no la *veracidad* del contenido. Un contrato estricto (`min_length=1`) sobre un texto que no tiene la información empuja al modelo a inventarla. Si el caso de uso lo necesita, la alternativa es permitir la lista vacía, o agregar un campo como `confianza`.

**2. Respuesta truncada** (`python main.py --demo-truncado`, con `max_tokens=40`):

```text
INFO    pipeline: Procesando texto (376 caracteres) con groq/openai/gpt-oss-20b
WARNING pipeline: Intento fallido (BadRequestError): Error code: 400 - {... 'code': 'tool_use_failed' ...}
INFO    pipeline: Reintento #1: se vuelve a llamar al modelo
WARNING pipeline: Intento fallido (BadRequestError): Error code: 400 - {... 'code': 'tool_use_failed' ...}
ERROR   pipeline: Extracción fallida tras 2 intento(s) (reintentos agotados): ...
[truncado] ✖ Error controlado tras 2 intento(s)
```

Con Groq, una tool call cortada no llega como `finish_reason=length`: el servidor la valida y la rechaza con `400 tool_use_failed`. Por eso el predicado de reintento reconoce ese código. Con OpenAI, en modo `json_schema`, el SDK lanza `LengthFinishReasonError`. Con Anthropic, la respuesta llega con `stop_reason=max_tokens` y la detecta `_validar_salida`. Los tres casos se reintentan y están cubiertos por los tests.

**3. API key inválida.** El error `401` no se reintenta: `Extracción fallida tras 1 intento(s) (error no reintentable)`.

## Tests

```bash
pytest
```

32 tests con un modelo falso, sin consumo de tokens. Cubren:

- las restricciones del schema;
- que el prompt reciba el texto y las instrucciones de formato;
- la recuperación ante `finish_reason=length`, `LengthFinishReasonError` de OpenAI, `stop_reason=max_tokens`, JSON inválido, respuesta sin estructura, `OutputParserException` y `tool_use_failed`;
- que se acepte `parsed` como dict (se revalida contra el schema);
- que se agoten los reintentos;
- que no se reintenten los errores definitivos y que `ExtraccionFallidaError` conserve la causa original;
- que llamadas concurrentes lleven cuentas de intentos separadas;
- que un texto vacío no llame al modelo;
- los errores de configuración: API key faltante, variable con valor inválido, paquete del proveedor faltante o que falla al importarse;
- que ninguna API key aparezca en los mensajes de error ni en los tracebacks.

## Estructura

```text
schemas.py        # Contrato Pydantic: ExtraccionTecnica, NivelCriticidad
config.py         # Settings desde .env (patrón del Módulo 1)
chain.py          # build_model, build_prompt, build_chain (LCEL + with_retry), process_text
main.py           # Mini-script de prueba asíncrono
tests/            # Tests del schema y de la cadena (modelo falso)
```

## Notas

- **Anthropic:** no se envía `temperature`, igual que en el Módulo 1.
- **Groq + `gpt-oss-20b`:** es un modelo de razonamiento, así que una parte de `max_tokens` se va en el razonamiento interno. Con valores muy bajos la respuesta se corta. Por eso el default es 2048.
- Las rutas de OpenAI y Anthropic se cubren con la misma cadena y los mismos tests, pero se probaron en vivo sólo contra Groq.
