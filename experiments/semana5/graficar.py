"""Gráfico clave de la semana 5: Recall@k vs k (holdout de validación).

Lee `resultados/recall_vs_k.csv` (generado por `ejecutar.py`) y guarda
`resultados/recall_vs_k.png`. Uso: python experiments/semana5/graficar.py
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

ETIQUETAS = {
    "B0": "Baseline: BM25",
    "V1": "V1: + raíces",
    "V2": "V2: + híbrido LSA",
    "V3": "V3: + enrutador",
}
COLORES = {"B0": SERIE[0], "V1": SERIE[1], "V2": SERIE[2], "V3": SERIE[3]}
ESTILOS = {"B0": "--", "V1": "-", "V2": "-", "V3": "-"}


def main() -> None:
    estilo()
    df = pd.read_csv(RES / "recall_vs_k.csv")
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    finales = []
    for v in ["B0", "V1", "V2", "V3"]:
        ax.plot(df["k"], df[v], ESTILOS[v], color=COLORES[v], lw=2.4 if v == "V3" else 1.8,
                marker="o", ms=4, label=ETIQUETAS[v], zorder=3 if v == "V3" else 2)
        finales.append((df[v].iloc[-1], v))

    # Rótulos directos a la derecha, separados para que no se encimen.
    finales.sort()
    ultimo = -1.0
    for y, v in finales:
        y_txt = max(y, ultimo + 0.032)
        ax.annotate(f"{ETIQUETAS[v]}  {y:.2f}", xy=(10, y), xytext=(10.25, y_txt),
                    color=COLORES[v] if v != "V3" else "#8a5d00", fontsize=9, va="center",
                    fontweight="bold" if v == "V3" else "normal",
                    arrowprops=dict(arrowstyle="-", color=REJ, lw=0.6) if abs(y_txt - y) > 0.005 else None)
        ultimo = y_txt

    ax.axvline(5, color=TINTA_2, lw=0.9, ls=":", zorder=1)
    ax.text(5.08, 0.05, "k = 5 (contexto del RAG)", color=TINTA_2, fontsize=8.5)
    ax.set_xticks(range(1, 11))
    ax.set_xlim(0.8, 10.2)
    ax.set_ylim(0, 0.75)
    ax.set_xlabel("k (fichas recuperadas)")
    ax.set_ylabel("Recall@k")
    ax.set_title("Recall@k vs k — holdout de validación (72 preguntas respondibles)", pad=12)
    fig.text(0.0, -0.04, "V3 = V1 + enrutador de subtema (β = 2,0). Batería v4, corpus v5.1. "
             "Fichas esperadas anotadas por pregunta (1–3).", color=TINTA_2, fontsize=8)
    fig.savefig(RES / "recall_vs_k.png")
    print("Guardado:", RES / "recall_vs_k.png")


REJ = "#c9c8c3"

if __name__ == "__main__":
    main()
