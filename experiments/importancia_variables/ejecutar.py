"""Importancia de variables del buscador del chatbot (sprint «importancia de variables»).

Configuración analizada: la adoptada al cerrar el sprint anterior, es decir, el buscador V1 (BM25 +
palabras vacías + raíces, sin enrutador) sobre el corpus v5.1 (D0).

La importancia se estudia desde dos lados:

A. Variables de los DATOS (campos de la ficha). ¿Qué parte del texto indexado de cada ficha ayuda a
   encontrarla? Se usa ablación «quitar uno»: la importancia de un campo es cuánto baja el Recall@5
   cuando ese campo se retira del índice (en las 136 fichas a la vez). Se mide con la CV de 5
   particiones en desarrollo y, una sola vez, en el holdout de validación (bootstrap pareado).

B. Variables de las PREGUNTAS. ¿Qué características hacen que el buscador falle (Fallo@5 = ninguna
   ficha correcta entre las 5)? Se entrena un modelo explicativo SOLO con desarrollo (regresión
   logística estandarizada y bosque aleatorio), se valida con CV agrupada por familia, y la
   importancia se mide por permutación sobre el holdout de validación.
   Nota: «solapamiento con la ficha correcta» usa la clave (fichas esperadas); es una variable de
   diagnóstico, no disponible cuando el chatbot está en uso.

La evaluación final no se usa.
Uso: python experiments/importancia_variables/ejecutar.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

RAIZ = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(RAIZ / "src"), str(RAIZ / "experiments/feature_engineering")]
import ejecutar as fe  # noqa: E402
from rag.metricas import diferencia_pareada  # noqa: E402
from rag.preprocesamiento import tokens_raices  # noqa: E402

SEMILLA, PRINCIPAL = 20261015, "recall@5"
CFG = dict(fe.VARIANTES["V1"])
DIR = Path(__file__).resolve().parent
R = DIR / "resultados"

# ---------------------------------------------------------------- A. campos de la ficha
PREFIJOS = {"relaciones": "Relaciones:", "condiciones": "Condiciones:", "error_frecuente": "Error frecuente:",
            "problema (ejemplos)": "Problema:", "resolucion (ejemplos)": "Resolución:", "respuesta (ejemplos)": "Respuesta:"}
CAMPOS = ["titulo", "contenido", *PREFIJOS]


def campo_de_linea(i: int, linea: str) -> str:
    if i == 0:
        return "titulo"
    for campo, pref in PREFIJOS.items():
        if linea.startswith(pref):
            return campo
    return "contenido"


def sin_campo(corpus: list[dict], campo: str) -> list[dict]:
    salida = []
    for c in corpus:
        lineas = [l for i, l in enumerate(c["index_text"].split("\n")) if campo_de_linea(i, l) != campo]
        salida.append({**c, "index_text": "\n".join(lineas) or c["titulo"]})
    return salida


def palabras_por_campo(corpus: list[dict]) -> dict[str, float]:
    tot = {k: 0 for k in CAMPOS}
    for c in corpus:
        for i, l in enumerate(c["index_text"].split("\n")):
            tot[campo_de_linea(i, l)] += len(re.findall(r"\w+", l))
    return {k: v / len(corpus) for k, v in tot.items()}


def cv_recall(corpus, dev):
    bus = fe.indice(CFG, corpus)[0]
    r = fe.evaluar(bus, dev).set_index("id")
    folds = StratifiedGroupKFold(5, shuffle=True, random_state=SEMILLA).split(dev, dev.subtema, dev.familia)
    return np.array([r.loc[dev.iloc[te].id, PRINCIPAL].mean() for _, te in folds])


def parte_a(corpus, dev, val):
    base_cv = cv_recall(corpus, dev)
    base_val = fe.evaluar(fe.indice(CFG, corpus)[0], val)
    pal = palabras_por_campo(corpus)
    filas = []
    for campo in CAMPOS:
        cs = sin_campo(corpus, campo)
        cvc = cv_recall(cs, dev)
        rv = fe.evaluar(fe.indice(CFG, cs)[0], val)
        d = diferencia_pareada(rv[PRINCIPAL], base_val[PRINCIPAL])      # completo − sin campo = importancia
        filas.append(dict(campo=campo, palabras_por_ficha=round(pal[campo], 1),
                          importancia_cv=round(float((base_cv - cvc).mean()), 4),
                          particiones_en_que_ayuda=int((base_cv > cvc + 1e-9).sum()),
                          recall5_val_sin_campo=round(float(rv[PRINCIPAL].mean()), 4),
                          importancia_val=round(d["diferencia"], 4),
                          ic95_val=f"[{d['ic95'][0]:+.3f}; {d['ic95'][1]:+.3f}]", p_val=d["p_bootstrap"],
                          importancia_val_practica=round(float(base_val.loc[base_val.componente == "practica", PRINCIPAL].mean()
                                                               - rv.loc[rv.componente == "practica", PRINCIPAL].mean()), 4),
                          importancia_val_teoria=round(float(base_val.loc[base_val.componente == "teoria", PRINCIPAL].mean()
                                                             - rv.loc[rv.componente == "teoria", PRINCIPAL].mean()), 4)))
    tabla = pd.DataFrame(filas).sort_values("importancia_cv", ascending=False)
    tabla.to_csv(R / "importancia_campos.csv", index=False)
    return tabla, dict(recall5_cv=round(float(base_cv.mean()), 4), recall5_val=round(float(base_val[PRINCIPAL].mean()), 4))


# ---------------------------------------------------------------- B. características de las preguntas
NIVEL = {"basico": 0, "intermedio": 1, "avanzado": 2}


def caracteristicas(q: pd.DataFrame, corpus: list[dict], resultados: pd.DataFrame) -> pd.DataFrame:
    texto = {c["id"]: set(tokens_raices(c["index_text"])) for c in corpus}
    filas = []
    for r in q.itertuples():
        tk = tokens_raices(r.enunciado)
        gold = set().union(*(texto[u] for u in r.unidades_relevantes))
        filas.append({"id": r.id,
                      "practica": int(r.componente == "practica"),
                      "nivel": NIVEL.get(r.nivel, 0),
                      "material_generado": int(r.origen == "generada"),
                      "palabras_enunciado": len(re.findall(r"\w+", r.enunciado)),
                      "cantidad_de_numeros": len(re.findall(r"\d+(?:[.,]\d+)?", r.enunciado)),
                      "fichas_esperadas": len(r.unidades_relevantes),
                      "cobertura_parcial": int(bool(r.cobertura_parcial)),
                      "parecida_a_ejemplo_del_corpus": int(len(r.similar_a_corpus or []) > 0),
                      "solapamiento_con_ficha_correcta": len(set(tk) & gold) / max(len(set(tk)), 1),
                      **{f"subtema_{s}": int(r.subtema == s) for s in ["MRU", "MRUV", "MVCL", "MPCL", "MCU"]}})
    X = pd.DataFrame(filas).set_index("id")
    X["fallo@5"] = resultados.set_index("id").loc[X.index, "fallo@5"].astype(int)
    return X


def parte_b(corpus, dev, val):
    bus = fe.indice(CFG, corpus)[0]
    Xd = caracteristicas(dev, corpus, fe.evaluar(bus, dev))
    Xv = caracteristicas(val, corpus, fe.evaluar(bus, val))
    feats = [c for c in Xd.columns if c != "fallo@5"]
    yd, yv = Xd.pop("fallo@5"), Xv.pop("fallo@5")
    modelos = {"regresion_logistica": make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)),
               "bosque_aleatorio": RandomForestClassifier(n_estimators=500, min_samples_leaf=5, random_state=SEMILLA)}
    auc, perm = {}, []
    for nombre, m in modelos.items():
        grupos = dev.set_index("id").loc[Xd.index, "familia"]
        aucs = []
        for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=SEMILLA).split(Xd, yd, grupos):
            m.fit(Xd.iloc[tr], yd.iloc[tr]); aucs.append(roc_auc_score(yd.iloc[te], m.predict_proba(Xd.iloc[te])[:, 1]))
        m.fit(Xd, yd)
        auc[nombre] = dict(auc_cv_desarrollo=round(float(np.mean(aucs)), 3), auc_cv_de=round(float(np.std(aucs)), 3),
                           auc_validacion=round(float(roc_auc_score(yv, m.predict_proba(Xv)[:, 1])), 3))
        pi = permutation_importance(m, Xv, yv, scoring="roc_auc", n_repeats=200, random_state=SEMILLA)
        for f, mu, sd in zip(feats, pi.importances_mean, pi.importances_std):
            perm.append(dict(modelo=nombre, variable=f, importancia_auc=round(float(mu), 4), de=round(float(sd), 4)))
    coef = modelos["regresion_logistica"][-1].coef_[0]
    coefs = pd.DataFrame(dict(variable=feats, coeficiente_estandarizado=np.round(coef, 3),
                              odds_ratio_por_1_de=np.round(np.exp(coef), 2))).sort_values("coeficiente_estandarizado")
    coefs.to_csv(R / "importancia_preguntas_coeficientes.csv", index=False)
    perm = pd.DataFrame(perm).sort_values(["modelo", "importancia_auc"], ascending=[True, False])
    perm.to_csv(R / "importancia_preguntas_permutacion.csv", index=False)
    # Lectura descriptiva sin modelo: tasa de fallo por grupo (desarrollo + validación)
    todo = pd.concat([Xd.assign(**{"fallo@5": yd}), Xv.assign(**{"fallo@5": yv})])
    todo["solapamiento (terciles)"] = pd.qcut(todo["solapamiento_con_ficha_correcta"], 3, labels=["bajo", "medio", "alto"])
    todo["subtema"] = todo[[c for c in todo if c.startswith("subtema_")]].idxmax(axis=1).str.replace("subtema_", "")
    grupos = []
    for g in ["practica", "subtema", "material_generado", "cobertura_parcial", "parecida_a_ejemplo_del_corpus", "solapamiento (terciles)"]:
        t = todo.groupby(g, observed=True)["fallo@5"].agg(["mean", "size"]).reset_index().rename(columns={g: "valor", "mean": "tasa_fallo@5", "size": "n"})
        t.insert(0, "variable", g); grupos.append(t)
    pd.concat(grupos).round(3).to_csv(R / "tasa_fallo_por_grupo.csv", index=False)
    return coefs, perm, auc, dict(n_desarrollo=len(Xd), n_validacion=len(Xv), fallo_desarrollo=round(float(yd.mean()), 3),
                                  fallo_validacion=round(float(yv.mean()), 3))


def main():
    corpus, q, fuente = fe.cargar(None)
    dev = q[q.particion == "desarrollo"].reset_index(drop=True)
    val = q[q.particion == "validacion"].reset_index(drop=True)
    R.mkdir(parents=True, exist_ok=True)
    print(f"Batería {fuente}: desarrollo {len(dev)} · validación {len(val)}")
    ta, base = parte_a(corpus, dev, val)
    print("\nA. Importancia de los campos de la ficha\n", ta.to_string(index=False))
    coefs, perm, auc, nb = parte_b(corpus, dev, val)
    print("\nB. Modelos explicativos del fallo:", auc)
    print(coefs.to_string(index=False))
    print(perm.to_string(index=False))
    json.dump(dict(bateria=fuente, configuracion="V1 (sin enrutador) + corpus v5.1 (D0)", linea_base=base, modelos=auc, **nb),
              open(R / "resumen.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
