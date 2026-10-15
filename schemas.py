"""Contrato de salida del pipeline: el objeto validado que devuelve la cadena."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator


class NivelCriticidad(StrEnum):
    BAJA = "baja"
    MEDIA = "media"
    ALTA = "alta"


class ExtraccionTecnica(BaseModel):
    """Entidades técnicas extraídas de un texto (arquitectura, log de error, incidente…)."""

    model_config = ConfigDict(extra="forbid")

    tecnologias: list[str] = Field(
        min_length=1,
        max_length=30,
        description=(
            "Tecnologías concretas mencionadas en el texto: lenguajes, frameworks, bases de "
            "datos, servicios cloud, colas, herramientas. Nombre propio y canónico, sin "
            "versiones ni descripciones (ej. 'PostgreSQL', no 'la base Postgres 15')."
        ),
    )
    nivel_de_criticidad: NivelCriticidad = Field(
        description=(
            "Impacto técnico: 'alta' si hay caída, pérdida de datos, seguridad o producción "
            "afectada; 'media' si hay degradación o riesgo; 'baja' si es informativo."
        ),
    )
    resumen_tecnico: str = Field(
        min_length=20,
        max_length=600,
        description="Resumen técnico de 1 a 3 oraciones, en español, sin inventar datos.",
    )

    @field_validator("tecnologias")
    @classmethod
    def _limpiar_tecnologias(cls, value: list[str]) -> list[str]:
        """Quita espacios, vacíos y duplicados (sin distinguir mayúsculas), conservando el orden."""
        vistas: set[str] = set()
        limpias: list[str] = []
        for tecnologia in (t.strip() for t in value):
            if tecnologia and tecnologia.casefold() not in vistas:
                vistas.add(tecnologia.casefold())
                limpias.append(tecnologia)
        if not limpias:
            raise ValueError("la lista de tecnologías no puede quedar vacía")
        return limpias

    @field_validator("nivel_de_criticidad", mode="before")
    @classmethod
    def _normalizar_criticidad(cls, value: object) -> object:
        # Los modelos a veces devuelven "Alta" o " alta ": se acepta, pero no "crítica".
        return value.strip().lower() if isinstance(value, str) else value

    @field_validator("resumen_tecnico")
    @classmethod
    def _limpiar_resumen(cls, value: str) -> str:
        return " ".join(value.split())
