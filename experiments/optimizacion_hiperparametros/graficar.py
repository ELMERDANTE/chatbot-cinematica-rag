"""Gráficos del sprint de optimización de hiperparámetros.

Uso: python experiments/optimizacion_hiperparametros/graficar.py
  → resultados/mapa_k1_b.png, efecto_marginal.png, recall_vs_k.png
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


def mapa():
    m = pd.read_csv(RES / "mapa_k1_b.csv", index_col=0)
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    im = ax.imshow(m.values, cmap="Blues", aspect="auto", origin="lower"); ax.grid(False)
    ax.set_xticks(range(m.shape[1])); ax.set_xticklabels(m.columns); ax.set_yticks(range(m.shape[0])); ax.set_yticklabels(m.index)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            v = m.values[i, j]
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=7.5, color="white" if v > m.values.mean() + 0.02 else "black")
    vi, vj = list(m.index).index(1.5), list(m.columns).index("0.75")
    ax.add_patch(plt.Rectangle((vj - .5, vi - .5), 1, 1, fill=False, ec=SERIE[1], lw=2))
    ax.set_xlabel("b (normalización por longitud)"); ax.set_ylabel("k1 (saturación del término)")
    ax.set_title("Recall@5 medio en la CV de desarrollo según k1 y b", pad=10)
    fig.colorbar(im, ax=ax, fraction=0.04)
    fig.text(0.0, -0.05, "Resto de hiperparámetros en su valor óptimo (raíces, título ×2, ejemplos sin enunciado). Recuadro naranja: k1 = 1,5 y b = 0,75 (vigentes).",
             color=TINTA_2, fontsize=8)
    fig.savefig(RES / "mapa_k1_b.png"); plt.close(fig)


def marginal():
    t = pd.read_csv(RES / "efecto_marginal.csv")
    nombres = {"preprocesamiento": "Preprocesamiento", "k1": "k1", "b": "b", "peso_titulo": "Peso del título", "ejemplos": "Ejemplos en el índice"}
    fig, axes = plt.subplots(1, 5, figsize=(11, 3.4), sharey=True)
    for ax, (h, g) in zip(axes, t.groupby("hiperparametro", sort=False)):
        ax.plot(range(len(g)), g.recall5_cv_medio, marker="o", color=SERIE[0])
        ax.set_xticks(range(len(g))); ax.set_xticklabels([str(v) for v in g.valor], rotation=45 if len(g) > 4 else 0, fontsize=8)
        ax.set_title(nombres.get(h, h), fontsize=10)
    axes[0].set_ylabel("Recall@5 medio (CV)")
    fig.suptitle("Efecto marginal de cada hiperparámetro (promedio sobre las 576 configuraciones)", y=1.04)
    fig.savefig(RES / "efecto_marginal.png"); plt.close(fig)


def curva():
    c = pd.read_csv(RES / "recall_vs_k.csv")
    etiq = {"vigente": "Vigente (k1 = 1,5; b = 0,75)", "solo_bm25": "Solo BM25 optimizado", "optimizada": "Optimizada"}
    fig, ax = plt.subplots(figsize=(7.8, 4.4))
    for i, n in enumerate(["vigente", "solo_bm25", "optimizada"]):
        ax.plot(c.k, c[n], "--" if n == "vigente" else "-", marker="o", ms=4, color=SERIE[i], lw=2.6 if n == "optimizada" else 1.6,
                label=f"{etiq[n]} ({c[n].iloc[4]:.2f} en k = 5)")
    ax.axvline(5, color=TINTA_2, ls=":", lw=1); ax.text(5.1, 0.05, "k = 5 (contexto del RAG)", color=TINTA_2, fontsize=8.5)
    ax.set_xticks(range(1, 11)); ax.set_ylim(0, 0.8); ax.set_xlabel("k (fichas recuperadas)"); ax.set_ylabel("Recall@k")
    ax.set_title("Recall@k vs k en validación (72 preguntas): antes y después de optimizar", pad=10); ax.legend(loc="lower right", fontsize=8.5)
    fig.savefig(RES / "recall_vs_k.png"); plt.close(fig)


if __name__ == "__main__":
    estilo(); mapa(); marginal(); curva(); print("Gráficos en", RES)
