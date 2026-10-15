"""Buscadores del componente de recuperación (la «R» de RAG).

- BM25: búsqueda léxica clásica.
- LSA: similitud semántica latente (TF-IDF + SVD), sustituto offline de un modelo
  de embeddings; se reemplazará por un modelo multilingüe en la siguiente fase.
- Híbrido: combinación lineal de BM25 y LSA, con las puntuaciones normalizadas por consulta.

Regla anti-fuga: `ajustar()` recibe SOLO el corpus. El vocabulario, el IDF y la
descomposición SVD nunca ven las preguntas de la batería.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from .preprocesamiento import PREPROCESADORES


def _minmax(x: np.ndarray) -> np.ndarray:
    rango = x.max() - x.min()
    return (x - x.min()) / rango if rango > 0 else np.zeros_like(x)


@dataclass
class Buscador:
    preprocesamiento: str = "basico"
    k1: float = 1.5
    b: float = 0.75
    hibrido: bool = False
    alpha: float = 0.5          # peso de BM25 en el híbrido
    dims_lsa: int = 100
    beta_subtema: float = 0.0   # peso del enrutador de subtema (0 = sin enrutador)
    semilla: int = 20261015
    ids: list = field(default_factory=list, repr=False)

    def ajustar(self, ids: list[str], textos: list[str], subtemas: list[str] | None = None, enrutador=None) -> "Buscador":
        """Construye el índice con el corpus (única fuente de ajuste del índice).

        `enrutador` (opcional) ya viene entrenado fuera, solo con datos de entrenamiento.
        """
        tok = PREPROCESADORES[self.preprocesamiento]
        self.ids = list(ids)
        self._sub = list(subtemas or [])
        self._enr = enrutador
        self._tok = tok
        self._bm25 = BM25Okapi([tok(t) for t in textos], k1=self.k1, b=self.b)
        if self.hibrido:
            self._tfidf = TfidfVectorizer(analyzer=tok, sublinear_tf=True)
            X = self._tfidf.fit_transform(textos)
            d = min(self.dims_lsa, X.shape[1] - 1, X.shape[0] - 1)
            self._svd = TruncatedSVD(n_components=d, random_state=self.semilla)
            self._docs = normalize(self._svd.fit_transform(X))
        return self

    def puntuar(self, consulta: str) -> np.ndarray:
        s = np.asarray(self._bm25.get_scores(self._tok(consulta)), dtype=float)
        if self.hibrido:
            q = normalize(self._svd.transform(self._tfidf.transform([consulta])))
            s_lsa = (self._docs @ q.T).ravel()
            s = self.alpha * _minmax(s) + (1 - self.alpha) * _minmax(s_lsa)
        if self._enr is not None and self.beta_subtema > 0:
            p = self._enr.proba(consulta)
            s = _minmax(s) + self.beta_subtema * np.array([p.get(t, 0.0) for t in self._sub])
        return s

    def buscar(self, consulta: str, k: int = 10) -> list[tuple[str, float]]:
        s = self.puntuar(consulta)
        orden = np.argsort(-s, kind="stable")[:k]
        return [(self.ids[i], float(s[i])) for i in orden]
