"""Gráfico clave del sprint de variantes de datos: Recall@k vs k (holdout de validación).

Uso: python experiments/variantes_datos/graficar.py   → resultados/recall_vs_k.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "src"))
from eda_utils import SERIE, TINTA_2, estilo  # noqa: E402

RES = Path(__file__).resolve().parent / "resultados"
ETIQ = {"D0": "D0 · corpus v5.1 (baseline)", "D1": "D1 · sin campos estructurados", "D2": "D2 · + problemas de Ágora"}
COL = {"D0": SERIE[0], "D1": SERIE[1], "D2": SERIE[2]}


def main() -> None:
    estilo()
    df = pd.read_csv(RES / "recall_vs_k.csv")
    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    for v in ["D0", "D1", "D2"]:
        ax.plot(df["k"], df[v], "--" if v == "D0" else "-", color=COL[v], marker="o", ms=4, lw=2.6 if v == "D2" else 1.8, zorder=3 if v == "D2" else 2)
    fin = sorted((df[v].iloc[-1], v) for v in ["D0", "D1", "D2"])
    ult = -1.0
    for y, v in fin:
        yt = max(y, ult + 0.035)
        ax.annotate(f"{ETIQ[v]}  {y:.2f}", xy=(10, y), xytext=(10.25, yt), va="center", fontsize=9,
                    color="#0f7a54" if v == "D2" else COL[v], fontweight="bold" if v == "D2" else "normal")
        ult = yt
    ax.axvline(5, color=TINTA_2, lw=0.9, ls=":")
    ax.text(5.08, 0.05, "k = 5 (contexto del RAG)", color=TINTA_2, fontsize=8.5)
    ax.set_xticks(range(1, 11)); ax.set_xlim(0.8, 10.2); ax.set_ylim(0, 0.85)
    ax.set_xlabel("k (fichas recuperadas)"); ax.set_ylabel("Recall@k")
    ax.set_title("Recall@k vs k — variantes de datos (buscador fijo V3, 72 preguntas de validación)", pad=12)
    fig.text(0.0, -0.04, "Solo cambia el corpus indexado; el buscador (raíces + enrutador de subtema) es el mismo en las tres variantes.",
             color=TINTA_2, fontsize=8)
    fig.savefig(RES / "recall_vs_k.png"); print("Guardado:", RES / "recall_vs_k.png")


if __name__ == "__main__":
    main()
