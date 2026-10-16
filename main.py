"""Mini-script de prueba asíncrono del pipeline de extracción.

Uso:
    python main.py                          # corre los 3 ejemplos (arquitectura, log, texto ambiguo)
    python main.py --texto "..."            # analiza un texto propio
    python main.py --provider anthropic     # cambia de proveedor (necesita su API key)
    python main.py --concurrente            # corre los ejemplos en paralelo con asyncio.gather
    python main.py --demo-truncado          # fuerza max_tokens bajo para ver detección y reintentos
    python main.py -q                       # sin logs, sólo los resultados
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import time

from pydantic import ValidationError

from chain import ExtraccionFallidaError, ExtractionPipeline
from config import Settings

EJEMPLOS = {
    "arquitectura": (
        "El backend es una API REST en FastAPI desplegada en Kubernetes sobre AWS EKS. "
        "Usa Redis como caché de sesiones y PostgreSQL como base principal, con migraciones "
        "en Alembic. Las tareas asíncronas corren en Celery con RabbitMQ como broker. En los "
        "picos de tráfico la latencia p99 sube a 4 s porque Redis se queda sin memoria y "
        "todas las lecturas terminan pegándole a PostgreSQL."
    ),
    "log_de_error": (
        "2026-10-14 03:12:55 ERROR [payments-worker] sqlalchemy.exc.OperationalError: "
        "(psycopg2.OperationalError) connection to server at \"db-prod.internal\" (10.0.3.12), "
        "port 5432 failed: FATAL: remaining connection slots are reserved for non-replication "
        "superuser connections\n"
        "2026-10-14 03:12:55 ERROR [payments-worker] Task charge_customer[8f2c] raised "
        "unexpected: OperationalError — retry 5/5 exhausted, message moved to DLQ "
        "(kafka topic payments.dlq)"
    ),
    "ambiguo": (
        "Ayer algo andaba lento, creo que era la base o quizás el cache, no sé bien. "
        "Alguien reinició el contenedor y pareció mejorar, aunque nadie revisó los logs."
    ),
}


def header(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


async def analizar(pipeline: ExtractionPipeline, nombre: str, texto: str) -> None:
    start = time.perf_counter()
    try:
        resultado = await pipeline.process_text(texto)
    except ExtraccionFallidaError as exc:
        print(f"\n[{nombre}] ✖ Error controlado tras {exc.intentos} intento(s): {exc}")
        return
    except ValueError as exc:
        print(f"\n[{nombre}] ✖ Entrada inválida: {exc}")
        return
    ms = (time.perf_counter() - start) * 1000
    print(f"\n[{nombre}] ✔ objeto validado en {ms:.0f} ms:")
    print(resultado.model_dump_json(indent=2))


async def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--provider", choices=["groq", "openai", "anthropic"])
    parser.add_argument("--texto", help="texto propio a analizar")
    parser.add_argument("--concurrente", action="store_true", help="correr los ejemplos en paralelo")
    parser.add_argument(
        "--demo-truncado",
        action="store_true",
        help="max_tokens muy bajo para forzar finish_reason=length y ver los reintentos",
    )
    parser.add_argument("-q", "--quiet", action="store_true", help="ocultar los logs del pipeline")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # Silenciar los logs HTTP de los SDKs para que se lean los del pipeline.
    for ruidoso in ("httpx", "openai", "anthropic", "groq"):
        logging.getLogger(ruidoso).setLevel(logging.WARNING)

    overrides = {"max_tokens": 40, "max_attempts": 2} if args.demo_truncado else {}
    try:
        settings = Settings.from_env(provider=args.provider, **overrides)
    except ValidationError as exc:
        # Configuración inválida (p.ej. falta la API key): se informa sin traceback.
        mensajes = "; ".join(e["msg"].removeprefix("Value error, ") for e in exc.errors())
        raise SystemExit(f"Configuración inválida: {mensajes}. Revisá el archivo .env.") from None

    if args.demo_truncado:
        header(f"Demo: max_tokens={settings.max_tokens} → respuesta cortada → reintento → error controlado")
        await analizar(ExtractionPipeline(settings), "truncado", EJEMPLOS["arquitectura"])
        return

    pipeline = ExtractionPipeline(settings)
    textos = {"texto": args.texto} if args.texto else EJEMPLOS

    if args.concurrente:
        header(f"{len(textos)} textos en paralelo con asyncio.gather")
        start = time.perf_counter()
        await asyncio.gather(*(analizar(pipeline, n, t) for n, t in textos.items()))
        print(f"\nTiempo total: {(time.perf_counter() - start) * 1000:.0f} ms")
        return

    for nombre, texto in textos.items():
        header(f"[{nombre}] {texto[:80]}…")
        await analizar(pipeline, nombre, texto)


if __name__ == "__main__":
    asyncio.run(main())
