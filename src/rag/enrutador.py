"""Enrutador de subtema: clasificador que estima P(subtema | consulta).

Se entrena con las fichas del corpus (etiquetadas por subtema) y con las preguntas
de las particiones de ENTRENAMIENTO. Nunca ve las preguntas con las que se evalúa.
"""
from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from .preprocesamiento import tokens_raices


class Enrutador:
    def __init__(self, C: float = 5.0, semilla: int = 20261015):
        self.modelo = make_pipeline(TfidfVectorizer(analyzer=tokens_raices, sublinear_tf=True, ngram_range=(1, 1)),
                                    LogisticRegression(C=C, max_iter=2000, random_state=semilla))

    def ajustar(self, textos: list[str], subtemas: list[str]) -> "Enrutador":
        self.modelo.fit(textos, subtemas)
        self.clases = list(self.modelo.classes_)
        return self

    def proba(self, consulta: str) -> dict[str, float]:
        p = self.modelo.predict_proba([consulta])[0]
        return dict(zip(self.clases, p))
