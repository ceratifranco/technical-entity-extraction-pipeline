# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Pre-entrega 2 del curso de AI Engineering: un pipeline LCEL que recibe texto técnico (arquitectura, logs) y devuelve un `ExtraccionTecnica` validado con Pydantic. Código, comentarios, mensajes de error, commits y README están en español rioplatense (voseo); mantener ese idioma.

## Comandos

No hay `python` en el PATH y el `python3` del sistema no es la versión del proyecto (3.12): usar siempre el intérprete del venv.

```bash
.venv/bin/python -m pytest                                   # toda la suite (no llama a ninguna API)
.venv/bin/python -m pytest tests/test_chain.py::test_nombre  # un test
.venv/bin/python -m pytest -k reintenta                      # por nombre
.venv/bin/python main.py -q --texto "..."                    # CLI; sin --texto corre 3 ejemplos (llamadas pagas/con cuota)
.venv/bin/python main.py --demo-truncado                     # fuerza max_tokens=40 para ver los reintentos
```

`pytest.ini` ya define `asyncio_mode = auto` y `pythonpath = .`: los tests `async def` no necesitan decorador. No hay linter ni formateador configurado.

## Arquitectura

Flujo (`chain.py`): `build_prompt() | _registrar_intento | model.with_structured_output(ExtraccionTecnica, include_raw=True) | _validar_salida`, envuelto en `.with_retry(retry_if_exception_type=_debe_reintentar)`.

- **`include_raw=True` es deliberado**: deja ver `finish_reason`/`stop_reason` y convierte los errores de parseo en `parsing_error` en vez de cortar la cadena. `_validar_salida` revisa en orden: respuesta truncada → `parsing_error` → `parsed is None`.
- **Qué se reintenta lo decide `es_reintentable()`**: sólo salida inválida (`SalidaInvalidaError` y subclases, `OutputParserException`, `ValidationError`, `LengthFinishReasonError` de OpenAI por nombre de clase, y los códigos `tool_use_failed`/`json_validate_failed` de Groq). Errores HTTP transitorios los reintenta el SDK (`max_retries`), no la cadena. Si se agrega un tipo de fallo, actualizar esta función, la tabla del README y los tests.
- **Contador de intentos en un `ContextVar`** (`_intentos`) para que las llamadas concurrentes (`asyncio.gather` en `--concurrente`) lleven cuentas separadas; `process_text` lo setea y resetea.
- **`ExtractionPipeline(settings, model=None)`** acepta un modelo inyectado: los tests usan `FakeModel`, que imita `with_structured_output(..., include_raw=True)` devolviendo respuestas o excepciones en orden. El `process_text()` a nivel de módulo es un atajo con singleton perezoso desde `.env`.
- **`schemas.py` es la única fuente de verdad del contrato**: las `description` de los campos viajan en el schema de la tool call y además generan `instrucciones_de_formato()` del prompt. Cambiar un campo cambia ambos.

### Errores

- `ExtraccionFallidaError` (con `.intentos`): reintentos agotados o error no reintentable; envuelve todo lo que falle dentro de `chain.ainvoke`.
- `EntradaVaciaError` y `ConfiguracionInvalidaError` heredan de `ValueError` (compatibilidad con quien capturaba el `ValidationError` de pydantic). `main.py` captura sólo estas excepciones propias, no `ValueError` genérico.
- `ConfiguracionInvalidaError` sale de `Settings.from_env()` y de `build_model()` (paquete del proveedor faltante sólo si es `ModuleNotFoundError` de `langchain_<provider>`; otro `ImportError` se muestra tal cual).

### Configuración y API keys (`config.py`)

- `ENV_VARS` mapea campo de `Settings` → variable de entorno; un setting nuevo necesita su entrada ahí para leerse del entorno y para que los errores nombren la variable.
- Las variables ya definidas en el entorno tienen prioridad sobre `.env` (`load_dotenv()` no las pisa); los `overrides` de `from_env` pisan ambas.
- No exponer API keys en mensajes ni tracebacks: `Settings` usa `hide_input_in_errors=True`, y los `ValidationError` de los chat models de LangChain se formatean con `describir_errores(exc, {})` y se relanzan `from None` (su `input_value` puede incluir la key). No usar `str(exc)` de esos errores en mensajes.
- Anthropic no recibe `temperature` (igual que el cliente del Módulo 1).

## Tests

Los tests de configuración usan el fixture `entorno_limpio` (en `tests/test_chain.py`), que anula `load_dotenv` y borra las variables de `ENV_VARS`, para no depender del `.env` real. Los que necesitan un paquete de proveedor usan `pytest.importorskip`. Al agregar comportamiento, mantener al día las secciones del README que lo documentan (tablas de errores y la lista de qué cubren los tests).
