"""Gráficos del sprint de importancia de variables.

Uso: python experiments/importancia_variables/graficar.py
  → resultados/importancia_campos.png y resultados/importancia_preguntas.png
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
NOMBRE = {"contenido": "Contenido", "titulo": "Título", "condiciones": "Condiciones", "relaciones": "Relaciones",
          "error_frecuente": "Error frecuente", "problema (ejemplos)": "Enunciado del ejemplo (fichas E)",
          "resolucion (ejemplos)": "Resolución del ejemplo (fichas E)", "respuesta (ejemplos)": "Respuesta del ejemplo (fichas E)",
          "solapamiento_con_ficha_correcta": "Solapamiento con la ficha correcta*", "cantidad_de_numeros": "Cantidad de números",
          "palabras_enunciado": "Longitud del enunciado", "parecida_a_ejemplo_del_corpus": "Parecida a un ejemplo del corpus",
          "practica": "Es de práctica", "cobertura_parcial": "Tema sin ficha específica", "material_generado": "Generada con IA",
          "fichas_esperadas": "N.º de fichas esperadas", "nivel": "Nivel"}


def campos():
    t = pd.read_csv(RES / "importancia_campos.csv").sort_values("importancia_val")
    fig, ax = plt.subplots(figsize=(7.8, 4.2))
    y = range(len(t))
    col = [SERIE[2] if v > 0 else SERIE[1] for v in t.importancia_val]
    ax.barh(y, t.importancia_val, color=col, height=0.6, label="Holdout de validación")
    ax.scatter(t.importancia_cv, y, color="black", zorder=3, s=18, label="Media de la CV en desarrollo")
    ax.set_yticks(list(y)); ax.set_yticklabels([NOMBRE.get(c, c) for c in t.campo])
    ax.axvline(0, color=TINTA_2, lw=0.8)
    ax.set_xlabel("Importancia = Recall@5 con el campo − Recall@5 sin el campo")
    ax.set_title("¿Qué parte de la ficha ayuda a encontrarla? (ablación «quitar uno», buscador V1)", pad=10)
    ax.legend(loc="lower right", fontsize=8)
    fig.text(0.0, -0.05, "Verde: quitar el campo empeora la búsqueda (el campo ayuda). Naranja: quitarlo la mejora (el campo estorba).",
             color=TINTA_2, fontsize=8)
    fig.savefig(RES / "importancia_campos.png"); plt.close(fig)


def preguntas():
    p = pd.read_csv(RES / "importancia_preguntas_permutacion.csv")
    p = p[p.modelo == "regresion_logistica"].sort_values("importancia_auc")
    p = p[~p.variable.str.startswith("subtema_")]
    fig, ax = plt.subplots(figsize=(7.8, 4.2))
    y = range(len(p))
    ax.barh(y, p.importancia_auc, xerr=p.de, color=SERIE[0], height=0.6, error_kw=dict(ecolor=TINTA_2, lw=0.8))
    ax.set_yticks(list(y)); ax.set_yticklabels([NOMBRE.get(v, v) for v in p.variable])
    ax.axvline(0, color=TINTA_2, lw=0.8)
    ax.set_xlabel("Caída del AUC al permutar la variable (validación, 200 repeticiones)")
    ax.set_title("¿Qué hace que el buscador falle en una pregunta? (regresión logística)", pad=10)
    fig.text(0.0, -0.05, "* Variable de diagnóstico: usa la clave de fichas esperadas; no está disponible cuando el chatbot está en uso. "
             "Subtemas omitidos del gráfico (importancia < 0,02).", color=TINTA_2, fontsize=8)
    fig.savefig(RES / "importancia_preguntas.png"); plt.close(fig)


if __name__ == "__main__":
    estilo(); campos(); preguntas(); print("Gráficos en", RES)
