"""Experimentos A/B centrados en los DATOS (sprint: informe de EDA + feature engineering + variantes).

El buscador queda FIJO en V1 (BM25 con palabras vacías + raíces, SIN enrutador de subtema).
Siguiendo la observación de la docente se retiró el enrutador de V3: su aporte frente a V1 no es
significativo y agrega un clasificador entrenado aparte que depende de los 5 subtemas de la batería.
Lo único que cambia entre variantes es el corpus que se indexa:

- D0 baseline : corpus v5.1 tal cual (título + contenido + ecuaciones + error frecuente).
- D1          : corpus v5.1 sin los campos estructurados (solo título + contenido).
- D2          : corpus v5.1 + problemas reales de Ágora de la reserva (no están en la batería),
                agregados al texto indexado de las fichas que usan (expansión de documentos).
                La expansión solo se usa para BUSCAR; al LLM se le entregará la ficha original.

Protocolo idéntico al avance de feature engineering: validación cruzada de 5 particiones en
desarrollo (estratificada por subtema, agrupada por familia) + holdout de validación.
La evaluación final no se usa.

Prueba adicional (no adoptada): el enrutador de subtema (V3) sobre D0 y D2, para documentar cuánto
aportaba antes de retirarlo → resultados/prueba_enrutador.csv.

Uso: python experiments/variantes_datos/ejecutar.py
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import StratifiedGroupKFold

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "src")); sys.path.insert(0, str(RAIZ / "experiments/feature_engineering"))
import ejecutar as fe  # noqa: E402  (funciones del avance anterior: cargar, indice, evaluar, entrenar_enrutador, elegir_beta)
from eda_utils import TOKENS_POR_PALABRA, palabras  # noqa: E402
from rag.metricas import diferencia_pareada, ic_bootstrap  # noqa: E402

SALIDA = RAIZ / "experiments/variantes_datos"
SEMILLA, K_MAX, PRINCIPAL = 20261015, 10, "recall@5"

VARIANTES = {
    "D0": dict(nombre="D0 · Baseline: corpus v5.1", cambio="—"),
    "D1": dict(nombre="D1 · Sin campos estructurados", cambio="Se indexa solo título + contenido (sin ecuaciones, condiciones ni error frecuente)"),
    "D2": dict(nombre="D2 · + Problemas reales de Ágora", cambio="Se agregan al texto indexado de cada ficha hasta 5 problemas de la reserva de Ágora que la usan"),
}


def texto_sin_campos(c: dict) -> str:
    return "\n".join(l for l in c["index_text"].split("\n") if not l.startswith(("Relaciones:", "Condiciones:", "Error frecuente:")))


def corpus_variante(nombre: str, corpus: list[dict], umbral: float = 0.45) -> list[dict]:
    if nombre == "D0":
        return corpus
    if nombre == "D1":
        return [{**c, "index_text": texto_sin_campos(c)} for c in corpus]
    exp = {json.loads(l)["ficha"]: json.loads(l)["problemas"]
           for l in open(RAIZ / "data/corpus/variantes/expansion_agora.jsonl", encoding="utf-8")}
    exp = {f: [p for p in ps if p["jaccard_max_bateria"] <= umbral] for f, ps in exp.items()}
    return [{**c, "index_text": c["index_text"] + ("\nProblemas tipo de la academia: " + " | ".join(p["enunciado"] for p in exp[c["id"]])
                                                   if exp.get(c["id"]) else "")} for c in corpus]


CFG = dict(fe.VARIANTES["V1"])      # buscador fijo: palabras vacías + raíces, sin enrutador
CFG_V3 = dict(fe.VARIANTES["V3"])   # solo para la prueba adicional del enrutador (no adoptado)


def cv(corpus, dev):
    """CV de 5 particiones (misma semilla y partición que el avance anterior)."""
    filas = []
    for f, (tr, te) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=SEMILLA).split(dev, dev.subtema, dev.familia), 1):
        dtr, dte = dev.iloc[tr], dev.iloc[te]
        assert not set(dtr.familia) & set(dte.familia)
        m = fe.evaluar(fe.indice(CFG, corpus)[0], dte)
        b0 = fe.evaluar(fe.indice(fe.VARIANTES["B0"], corpus)[0], dte)
        filas.append(dict(fold=f, **{"recall@5": m[PRINCIPAL].mean(), "recall@10": m["recall@10"].mean(),
                                                  "mrr@10": m["mrr@10"].mean(), "recall@5 (BM25 base)": b0[PRINCIPAL].mean()}))
    return pd.DataFrame(filas)


def main():
    corpus0, q, fuente = fe.cargar(None)
    dev = q[q.particion == "desarrollo"].reset_index(drop=True)
    val = q[q.particion == "validacion"].reset_index(drop=True)
    commit = subprocess.run(["git", "-C", str(RAIZ), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    fecha = dt.datetime.now(dt.timezone(dt.timedelta(hours=-5))).isoformat(timespec="seconds")
    print(f"Batería {fuente}: desarrollo {len(dev)} · validación {len(val)}")

    res, cvs, extra = {}, {}, {}
    for v in VARIANTES:
        corpus = corpus_variante(v, corpus0)
        cvdf = cv(corpus, dev); cvdf.insert(0, "variante", v); cvs[v] = cvdf
        conf = {"experimento": "variantes-datos", "variante": v, **CFG, "enrutador": False, "corpus": VARIANTES[v]["nombre"],
                "k_max": K_MAX, "semilla": SEMILLA, "bateria": "bateria_v4", "particion_evaluada": "validacion"}
        texto_conf = yaml.safe_dump(conf, allow_unicode=True, sort_keys=False)
        chash = hashlib.sha256(texto_conf.encode()).hexdigest()[:12]; run_id = f"dv-{v}-{chash}"
        carpeta = SALIDA / "runs" / run_id; carpeta.mkdir(parents=True, exist_ok=True)
        (carpeta / "config.yaml").write_text(texto_conf, encoding="utf-8")
        construcciones = [fe.indice(CFG, corpus) for _ in range(5)]
        bus, t_idx = construcciones[0][0], float(np.median([t for _, t in construcciones]))
        log = open(carpeta / "log.jsonl", "w", encoding="utf-8")

        def registrar(r, top, m, lat):
            log.write(json.dumps(dict(run_id=run_id, config_hash=chash, commit=commit, fecha=fecha, consulta_id=r.id,
                                      consulta_texto=r.enunciado if r.texto_publicable else None,
                                      topk=[dict(chunk_id=d, score=round(s, 4), rango=i) for i, (d, s) in enumerate(top, 1)],
                                      relevantes=list(r.unidades_relevantes), recall_5=m["recall@5"], mrr_10=round(m["mrr@10"], 4),
                                      latencia_ms=round(lat, 3)), ensure_ascii=False) + "\n")
        r = fe.evaluar(bus, val, registrar); log.close(); res[v] = r
        b0 = fe.evaluar(fe.indice(fe.VARIANTES["B0"], corpus)[0], val)
        pal = [palabras(c["index_text"]) for c in corpus]
        extra[v] = dict(indice_ms=round(t_idx, 1), consulta_ms=round(r.latencia_ms.mean(), 2),
                        palabras_indexadas_media=round(float(np.mean(pal)), 1),
                        recall5_bm25_base=float(b0[PRINCIPAL].mean()))
        resumen = {k: float(r[k].mean()) for k in r.columns if k.startswith(("recall", "precision", "mrr", "ndcg", "fallo", "acierto"))}
        (carpeta / "metricas.json").write_text(json.dumps(dict(run_id=run_id, n=len(r), **resumen, **extra[v]), indent=1), encoding="utf-8")
        print(f"{v}: Recall@5 = {resumen['recall@5']:.3f} · run {run_id}")

    R = SALIDA / "resultados"; R.mkdir(exist_ok=True)
    filas = []
    for v, r in res.items():
        lo, hi = ic_bootstrap(r[PRINCIPAL])
        f = dict(variante=VARIANTES[v]["nombre"], cambio=VARIANTES[v]["cambio"], **{
            "Recall@5": r[PRINCIPAL].mean(), "IC95 Recall@5": f"[{lo:.3f}; {hi:.3f}]", "Recall@1": r["recall@1"].mean(),
            "Recall@10": r["recall@10"].mean(), "MRR@10": r["mrr@10"].mean(), "nDCG@5": r["ndcg@5"].mean(),
            "Fallo@5": r["fallo@5"].mean(), "Acierto de subtema@1": r["acierto_subtema@1"].mean(),
            "Recall@5 práctica": r.loc[r.componente == "practica", PRINCIPAL].mean(),
            "Recall@5 teoría": r.loc[r.componente == "teoria", PRINCIPAL].mean()}, **extra[v])
        if v != "D0":
            d = diferencia_pareada(res["D0"][PRINCIPAL], r[PRINCIPAL])
            f.update({"Δ Recall@5 vs D0": d["diferencia"], "IC95 Δ vs D0": f"[{d['ic95'][0]:+.3f}; {d['ic95'][1]:+.3f}]", "p vs D0": d["p_bootstrap"]})
            p = diferencia_pareada(res["D0"].loc[res["D0"].componente == "practica", PRINCIPAL], r.loc[r.componente == "practica", PRINCIPAL])
            f.update({"Δ práctica vs D0": p["diferencia"], "p práctica": p["p_bootstrap"]})
        filas.append(f)
    tabla = pd.DataFrame(filas); tabla.to_csv(R / "tabla_principal.csv", index=False)
    curva = pd.DataFrame({v: [r[f"recall@{k}"].mean() for k in range(1, K_MAX + 1)] for v, r in res.items()}, index=range(1, K_MAX + 1))
    curva.index.name = "k"; curva.to_csv(R / "recall_vs_k.csv")
    cvall = pd.concat(cvs.values()); cvall.to_csv(R / "cv_desarrollo.csv", index=False)
    cvres = cvall.groupby("variante")[["recall@5", "recall@10", "mrr@10", "recall@5 (BM25 base)"]].agg(["mean", "std"]).round(3)
    porsub = pd.DataFrame({v: r.groupby("subtema")[PRINCIPAL].mean() for v, r in res.items()}).round(3); porsub.to_csv(R / "recall5_por_subtema.csv")
    json.dump(dict(fecha=fecha, commit=commit, bateria=fuente, n_desarrollo=len(dev), n_validacion=len(val),
                   cv=json.loads(cvres.to_json()), extra=extra), open(R / "resumen.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    # Sensibilidad de D2: ¿la mejora se mantiene si se exige que los problemas agregados se parezcan MENOS a la batería?
    sens = []
    for umbral in (0.45, 0.40, 0.35, 0.30):
        corpus = corpus_variante("D2", corpus0, umbral)
        n = sum(c["index_text"].count("Problemas tipo") and (c["index_text"].split("Problemas tipo de la academia: ")[1].count(" | ") + 1) for c in corpus)
        r = fe.evaluar(fe.indice(CFG, corpus)[0], val)
        d = diferencia_pareada(res["D0"][PRINCIPAL], r[PRINCIPAL])
        sens.append(dict(umbral_jaccard=umbral, problemas_agregados=n, recall5=r[PRINCIPAL].mean(),
                         delta_vs_D0=d["diferencia"], p=d["p_bootstrap"]))
    sens = pd.DataFrame(sens); sens.to_csv(R / "sensibilidad_umbral_D2.csv", index=False); print(sens.round(3))
    # Prueba adicional: ¿cuánto aportaba el enrutador de subtema (V3) que se retiró?
    enr_filas = []
    for v in ("D0", "D2"):
        corpus = corpus_variante(v, corpus0)
        beta = fe.elegir_beta(corpus, dev)          # β elegido con CV anidada solo en desarrollo
        r3 = fe.evaluar(fe.indice({**CFG_V3, "beta_subtema": beta}, corpus, fe.entrenar_enrutador(corpus, dev))[0], val)
        d = diferencia_pareada(res[v][PRINCIPAL], r3[PRINCIPAL])
        g3 = []
        for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=SEMILLA).split(dev, dev.subtema, dev.familia):
            dtr, dte = dev.iloc[tr], dev.iloc[te]
            b = fe.elegir_beta(corpus, dtr)
            g3.append(fe.evaluar(fe.indice({**CFG_V3, "beta_subtema": b}, corpus, fe.entrenar_enrutador(corpus, dtr))[0], dte)[PRINCIPAL].mean())
        g1 = cvs[v]["recall@5"].to_numpy()
        enr_filas.append(dict(corpus=v, recall5_sin_enrutador=res[v][PRINCIPAL].mean(), recall5_con_enrutador=r3[PRINCIPAL].mean(),
                              beta=beta, delta=d["diferencia"], ic95=f"[{d['ic95'][0]:+.3f}; {d['ic95'][1]:+.3f}]", p=d["p_bootstrap"],
                              cv_sin_enrutador=round(float(np.mean(g1)), 3), cv_con_enrutador=round(float(np.mean(g3)), 3),
                              particiones_cv_gana_enrutador=int(sum(np.array(g3) > g1 + 1e-9)),
                              acierto_subtema_sin=res[v]["acierto_subtema@1"].mean(), acierto_subtema_con=r3["acierto_subtema@1"].mean()))
    enr_df = pd.DataFrame(enr_filas); enr_df.to_csv(R / "prueba_enrutador.csv", index=False); print(enr_df.round(3).to_string())
    pd.set_option("display.width", 250)
    print(tabla.round(3).drop(columns=["cambio"]).to_string()); print(cvres)


if __name__ == "__main__":
    main()
