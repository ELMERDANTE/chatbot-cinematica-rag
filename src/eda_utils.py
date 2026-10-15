"""Utilidades del análisis exploratorio (EDA) del corpus y de la batería de evaluación.

Solo usa archivos publicados en el repositorio, de modo que el notebook puede
reproducirse desde un clon limpio.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
FIG = RAIZ / "results/eda/figuras"
TAB = RAIZ / "results/eda/tablas"

SUBTEMAS = ["MRU", "MRUV", "MVCL", "MPCL", "MCU"]
NIVELES = ["basico", "intermedio", "avanzado"]
PARTICIONES = ["desarrollo", "validacion", "evaluacion_final"]

# Paleta categórica validada (orden fijo, nunca cíclico) y tinta de texto.
SERIE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
SECUENCIAL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SUPERFICIE = "#fcfcfb"
TINTA = "#0b0b0b"
TINTA_2 = "#52514e"
REJILLA = "#e4e3df"

# Factor aproximado palabras → tokens para texto en español con tokenizadores BPE
# de LLM comerciales (estimación; el conteo exacto depende del modelo elegido).
TOKENS_POR_PALABRA = 1.6


def estilo() -> None:
    plt.rcParams.update({
        "figure.facecolor": SUPERFICIE, "axes.facecolor": SUPERFICIE, "savefig.facecolor": SUPERFICIE,
        "axes.edgecolor": REJILLA, "axes.labelcolor": TINTA_2, "xtick.color": TINTA_2, "ytick.color": TINTA_2,
        "text.color": TINTA, "axes.titlecolor": TINTA, "axes.titleweight": "bold", "axes.titlesize": 12,
        "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": REJILLA, "grid.linewidth": 0.8, "axes.axisbelow": True,
        "font.size": 10, "legend.frameon": False, "figure.dpi": 110, "savefig.dpi": 160,
        "savefig.bbox": "tight",
    })


def palabras(texto: str | None) -> int:
    return len(re.findall(r"\w+", texto or ""))


def cargar_corpus() -> pd.DataFrame:
    filas = [json.loads(l) for l in open(RAIZ / "data/corpus/processed/corpus_v5_1.jsonl", encoding="utf-8")]
    df = pd.DataFrame(filas)
    df["subtema"] = df["subtema_proyecto"]
    df["palabras"] = df["index_text"].map(palabras)
    df["tokens_est"] = (df["palabras"] * TOKENS_POR_PALABRA).round().astype(int)
    df["n_ecuaciones"] = df["ecuaciones"].map(len)
    df["n_errores"] = df["errores_frecuentes"].map(len)
    df["fuente_libro"] = df["fuente"].map(lambda fs: ", ".join(sorted({
        "Lumbreras" if "Lumbreras" in f["libro"] else "Giancoli" if "Giancoli" in f["libro"] else "Síntesis propia"
        for f in fs})) if fs else "Sin fuente")
    return df


def cargar_bateria() -> pd.DataFrame:
    df = pd.read_csv(RAIZ / "data/bateria/bateria_v4_metadatos.csv")
    df["unidades"] = df["unidades_relevantes"].fillna("").map(lambda s: [u for u in s.split(";") if u])
    return df


def cargar_inventario() -> pd.DataFrame:
    return pd.read_csv(RAIZ / "data/eda/inventario_fuentes.csv")


def guardar(fig, nombre: str) -> Path:
    FIG.mkdir(parents=True, exist_ok=True)
    ruta = FIG / f"{nombre}.png"
    fig.savefig(ruta)
    return ruta


def guardar_tabla(df: pd.DataFrame, nombre: str) -> Path:
    TAB.mkdir(parents=True, exist_ok=True)
    ruta = TAB / f"{nombre}.csv"
    df.to_csv(ruta, encoding="utf-8")
    return ruta


def barras_apiladas(ax, tabla: pd.DataFrame, colores: list[str], etiquetas: bool = True) -> None:
    """Barras apiladas con separación de 2 px (borde del color de superficie) y rótulos de valor."""
    base = pd.Series(0, index=tabla.index, dtype=float)
    for col, color in zip(tabla.columns, colores):
        vals = tabla[col]
        ax.bar(tabla.index, vals, bottom=base, color=color, edgecolor=SUPERFICIE, linewidth=2, width=0.62, label=col)
        if etiquetas:
            for x, (v, b) in enumerate(zip(vals, base)):
                if v >= 4:
                    ax.text(x, b + v / 2, f"{int(v)}", ha="center", va="center", fontsize=8.5, color=TINTA)
        base = base + vals
    for x, total in enumerate(base):
        ax.text(x, total + max(base) * 0.015, f"{int(total)}", ha="center", va="bottom", fontsize=9, color=TINTA_2)
