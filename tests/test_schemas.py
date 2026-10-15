import pytest
from pydantic import ValidationError

from schemas import ExtraccionTecnica, NivelCriticidad

RESUMEN = "API con caché en Redis y persistencia en PostgreSQL."


def test_objeto_valido():
    obj = ExtraccionTecnica(
        tecnologias=["FastAPI", "Redis"], nivel_de_criticidad="alta", resumen_tecnico=RESUMEN
    )
    assert obj.nivel_de_criticidad is NivelCriticidad.ALTA
    assert obj.model_dump(mode="json")["nivel_de_criticidad"] == "alta"


@pytest.mark.parametrize("tecnologias", [[], ["", "  "]])
def test_tecnologias_no_puede_estar_vacia(tecnologias):
    with pytest.raises(ValidationError):
        ExtraccionTecnica(
            tecnologias=tecnologias, nivel_de_criticidad="baja", resumen_tecnico=RESUMEN
        )


def test_tecnologias_se_limpian_y_deduplican():
    obj = ExtraccionTecnica(
        tecnologias=[" Redis ", "redis", "PostgreSQL"],
        nivel_de_criticidad="media",
        resumen_tecnico=RESUMEN,
    )
    assert obj.tecnologias == ["Redis", "PostgreSQL"]


def test_criticidad_normaliza_mayusculas_pero_rechaza_valores_fuera_del_enum():
    ok = ExtraccionTecnica(tecnologias=["Redis"], nivel_de_criticidad=" Alta ", resumen_tecnico=RESUMEN)
    assert ok.nivel_de_criticidad is NivelCriticidad.ALTA
    with pytest.raises(ValidationError):
        ExtraccionTecnica(tecnologias=["Redis"], nivel_de_criticidad="crítica", resumen_tecnico=RESUMEN)


@pytest.mark.parametrize("resumen", ["corto", "x" * 601])
def test_resumen_con_longitud_acotada(resumen):
    with pytest.raises(ValidationError):
        ExtraccionTecnica(tecnologias=["Redis"], nivel_de_criticidad="baja", resumen_tecnico=resumen)


def test_campos_extra_o_faltantes_fallan():
    with pytest.raises(ValidationError):
        ExtraccionTecnica.model_validate({"tecnologias": ["Redis"], "nivel_de_criticidad": "baja"})
    with pytest.raises(ValidationError):
        ExtraccionTecnica.model_validate(
            {"tecnologias": ["Redis"], "nivel_de_criticidad": "baja", "resumen_tecnico": RESUMEN, "x": 1}
        )
