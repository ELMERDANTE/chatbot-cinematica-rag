"""Métricas de recuperación por consulta y su agregación con intervalos bootstrap."""
from __future__ import annotations

import math

import numpy as np


def recall_at(rec: list[str], rel: set[str], k: int) -> float:
    return len(set(rec[:k]) & rel) / len(rel)


def precision_at(rec: list[str], rel: set[str], k: int) -> float:
    return len(set(rec[:k]) & rel) / k


def mrr_at(rec: list[str], rel: set[str], k: int = 10) -> float:
    for i, d in enumerate(rec[:k], 1):
        if d in rel:
            return 1 / i
    return 0.0


def ndcg_at(rec: list[str], rel: set[str], k: int = 5) -> float:
    dcg = sum(1 / math.log2(i + 1) for i, d in enumerate(rec[:k], 1) if d in rel)
    ideal = sum(1 / math.log2(i + 1) for i in range(1, min(len(rel), k) + 1))
    return dcg / ideal


def por_consulta(rec: list[str], rel: set[str], ks=range(1, 11)) -> dict:
    m = {f"recall@{k}": recall_at(rec, rel, k) for k in ks}
    m.update({"precision@5": precision_at(rec, rel, 5), "mrr@10": mrr_at(rec, rel, 10),
              "ndcg@5": ndcg_at(rec, rel, 5), "fallo@5": float(not set(rec[:5]) & rel)})
    return m


def ic_bootstrap(valores, n: int = 2000, semilla: int = 20261015) -> tuple[float, float]:
    """Intervalo de confianza del 95 % de la media (bootstrap por percentiles)."""
    v = np.asarray(valores, dtype=float)
    rng = np.random.default_rng(semilla)
    medias = v[rng.integers(0, len(v), (n, len(v)))].mean(axis=1)
    return float(np.percentile(medias, 2.5)), float(np.percentile(medias, 97.5))


def diferencia_pareada(a, b, n: int = 5000, semilla: int = 20261015) -> dict:
    """Diferencia media pareada (b − a), IC 95 % bootstrap y p bilateral aproximado."""
    d = np.asarray(b, float) - np.asarray(a, float)
    rng = np.random.default_rng(semilla)
    medias = d[rng.integers(0, len(d), (n, len(d)))].mean(axis=1)
    p = 2 * min((medias <= 0).mean(), (medias >= 0).mean())
    return {"diferencia": float(d.mean()), "ic95": (float(np.percentile(medias, 2.5)), float(np.percentile(medias, 97.5))),
            "p_bootstrap": float(min(p, 1.0))}
