"""Preprocesamiento de texto (el «feature set» del buscador).

Cada variante del experimento elige un preprocesamiento. Ninguna transformación
aprende de las preguntas: lo único que se ajusta con datos (vocabulario, IDF, LSA)
se ajusta SOLO con el corpus (ver recuperacion.py).
"""
from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

from nltk.stem.snowball import SpanishStemmer

# Lista propia de palabras vacías del español (artículos, preposiciones, pronombres,
# conectores y verbos auxiliares frecuentes en los enunciados). No incluye palabras
# con contenido físico («alto», «bajo», «vuelta», «tiempo», etc.).
PALABRAS_VACIAS = frozenset("""
a al algo algun alguna algunas alguno algunos ante antes aquel aquella aquellas aquellos aqui asi aun
bajo cada como con contra cual cuales cualquier cuando cuanto cuantos cuanta cuantas de del desde
donde dos durante e el ella ellas ello ellos en entre era eran es esa esas ese eso esos esta estaba
estan estar este esto estos fue fueron ha han hasta hay la las le les lo los mas me mi mientras muy
ni no nos o otra otras otro otros para pero poco por porque que quien quienes se sea ser si sin sobre
son su sus tal tambien tan tanto te tiene tienen todo todos tu un una unas uno unos y ya
determine halle hallar calcule calcular indique señale considere luego dicho dicha cierto cierta
""".split())

_STEM = SpanishStemmer()
_TOKEN = re.compile(r"[a-z0-9]+(?:[.,][0-9]+)?")


def normalizar(texto: str) -> str:
    """Minúsculas y sin tildes (la ñ se conserva como n)."""
    t = unicodedata.normalize("NFD", (texto or "").lower())
    return "".join(c for c in t if unicodedata.category(c) != "Mn")


@lru_cache(maxsize=50_000)
def _raiz(palabra: str) -> str:
    return _STEM.stem(palabra)


def tokens_basico(texto: str) -> list[str]:
    """Baseline: normalización mínima y separación en palabras."""
    return _TOKEN.findall(normalizar(texto))


def tokens_raices(texto: str) -> list[str]:
    """Variante 1: baseline + eliminación de palabras vacías + raíces (Snowball español)."""
    return [_raiz(t) for t in tokens_basico(texto) if t not in PALABRAS_VACIAS and not t.isdigit()]


PREPROCESADORES = {"basico": tokens_basico, "raices": tokens_raices}
