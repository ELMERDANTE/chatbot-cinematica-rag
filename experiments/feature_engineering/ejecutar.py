"""Feature engineering y experimentos A/B del componente de recuperación.

Protocolo
---------
1. Índice: se ajusta SOLO con el corpus v5.1 (las preguntas nunca se usan para ajustar).
2. Desarrollo (180 preguntas respondibles): validación cruzada de 5 particiones,
   estratificada por subtema y agrupada por familia (StratifiedGroupKFold). Sirve para
   estimar la estabilidad de cada variante y para elegir los hiperparámetros de V2
   dentro de las particiones de entrenamiento.
3. Validación (72 preguntas respondibles): holdout; cada variante se ejecuta una vez
   con su configuración final. Aquí se comparan las variantes (tabla principal).
4. Evaluación final (108 respondibles): NO se toca en esta etapa.

Variantes (un cambio por vez, en escalera):
- B0  baseline : BM25, normalización mínima.
- V1  B0 + palabras vacías y raíces (stemming en español).
- V2  V1 + búsqueda híbrida (BM25 + LSA).
- V3  V1 + enrutador de subtema (clasificador entrenado con el corpus y las preguntas
      de entrenamiento). Se construye sobre V1 porque V2 no la superó en desarrollo.

Uso: python experiments/feature_engineering/ejecutar.py [--bateria RUTA]
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import subprocess
import sys
import time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import StratifiedGroupKFold

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "src"))
from rag.metricas import diferencia_pareada, ic_bootstrap, por_consulta  # noqa: E402
from rag.enrutador import Enrutador  # noqa: E402
from rag.recuperacion import Buscador  # noqa: E402

SALIDA = RAIZ / "experiments/feature_engineering"
SEMILLA = 20261015
K_MAX = 10
VARIANTES = {
    "B0": dict(nombre="Baseline: BM25 (normalización mínima)", cambio="—", preprocesamiento="basico"),
    "V1": dict(nombre="V1: + palabras vacías y raíces", cambio="Preprocesamiento: palabras vacías + stemming español (Snowball)",
               preprocesamiento="raices"),
    "V2": dict(nombre="V2: + búsqueda híbrida BM25 + LSA", cambio="Puntuación: combinación de BM25 con similitud semántica LSA",
               preprocesamiento="raices", hibrido=True),
    "V3": dict(nombre="V3: V1 + enrutador de subtema", cambio="Prioriza las fichas del subtema que predice un clasificador entrenado solo con datos de entrenamiento",
               preprocesamiento="raices", base="V1"),
}
VARIANTES["V2"]["base"] = "V1"; VARIANTES["V1"]["base"] = "B0"
GRID_V3 = {"beta_subtema": [0.5, 1.0, 2.0, 4.0]}
GRID_V2 = {"alpha": [0.3, 0.5, 0.7], "dims_lsa": [50, 100]}
PRINCIPAL = "recall@5"


def cargar(ruta_bateria: Path | None):
    corpus = [json.loads(l) for l in open(RAIZ / "data/corpus/processed/corpus_v5_1.jsonl", encoding="utf-8")]
    privada = ruta_bateria or RAIZ / "data/bateria/privado/bateria_v4_completa.jsonl"
    if privada.exists():
        bat = [json.loads(l) for l in open(privada, encoding="utf-8")]; fuente = "completa (privada)"
    else:
        bat = [json.loads(l) for l in open(RAIZ / "data/bateria/bateria_v4_publica.jsonl", encoding="utf-8")]
        bat = [b for b in bat if b["enunciado"]]; fuente = "pública (sin las preguntas de libros protegidos)"
    q = pd.DataFrame([b for b in bat if b["respondible"]])
    assert set(q.particion) <= {"desarrollo", "validacion", "evaluacion_final"}
    return corpus, q, fuente


def entrenar_enrutador(corpus: list[dict], preguntas: pd.DataFrame) -> Enrutador:
    """Ajuste SOLO con el corpus y las preguntas de entrenamiento recibidas."""
    textos = [c["index_text"] for c in corpus] + list(preguntas.enunciado)
    etiquetas = [c["subtema_proyecto"] for c in corpus] + list(preguntas.subtema)
    return Enrutador().ajustar(textos, etiquetas)


def indice(cfg: dict, corpus: list[dict], enrutador=None) -> tuple[Buscador, float]:
    t0 = time.perf_counter()
    bus = Buscador(**{k: v for k, v in cfg.items() if k in Buscador.__dataclass_fields__ and k != "ids"}).ajustar(
        [c["id"] for c in corpus], [c["index_text"] for c in corpus], [c["subtema_proyecto"] for c in corpus], enrutador)
    return bus, (time.perf_counter() - t0) * 1000


def elegir_beta(corpus, entren: pd.DataFrame) -> float:
    """CV interna (3 particiones) dentro de los datos de entrenamiento para elegir β del enrutador."""
    cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=SEMILLA + 1)
    puntos = {b: [] for b in GRID_V3["beta_subtema"]}
    for tr, te in cv.split(entren, entren.subtema, entren.familia):
        enr = entrenar_enrutador(corpus, entren.iloc[tr])
        for b in puntos:
            bus, _ = indice({**VARIANTES["V3"], "beta_subtema": b}, corpus, enr)
            puntos[b].append(evaluar(bus, entren.iloc[te])[PRINCIPAL].mean())
    return max(puntos, key=lambda b: (np.mean(puntos[b]), -b))


def evaluar(bus: Buscador, q: pd.DataFrame, registrar=None) -> pd.DataFrame:
    filas = []
    for r in q.itertuples():
        t0 = time.perf_counter(); top = bus.buscar(r.enunciado, K_MAX); lat = (time.perf_counter() - t0) * 1000
        rec = [d for d, _ in top]; m = por_consulta(rec, set(r.unidades_relevantes))
        m["acierto_subtema@1"] = float(rec[0].split("-")[1] == r.subtema)
        m.update(id=r.id, subtema=r.subtema, componente=r.componente, nivel=r.nivel, origen=r.origen,
                 cobertura_parcial=bool(getattr(r, "cobertura_parcial", False)), latencia_ms=lat)
        filas.append(m)
        if registrar: registrar(r, top, m, lat)
    return pd.DataFrame(filas)


def validacion_cruzada(corpus, dev: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEMILLA)
    filas, elegidos = [], []
    cache = {}
    def res(nombre, cfg, sub):
        clave = (nombre, json.dumps(cfg, sort_keys=True))
        if clave not in cache: cache[clave] = evaluar(indice(cfg, corpus)[0], dev)
        return cache[clave].loc[cache[clave].id.isin(sub.id)]
    for f, (tr, te) in enumerate(cv.split(dev, dev.subtema, dev.familia), 1):
        dtr, dte = dev.iloc[tr], dev.iloc[te]
        assert not set(dtr.familia) & set(dte.familia), "familia partida entre particiones"
        for v, base in VARIANTES.items():
            cfg = dict(base)
            if v == "V3":  # enrutador entrenado y β elegido SOLO con la partición de entrenamiento
                beta = elegir_beta(corpus, dtr); cfg["beta_subtema"] = beta
                elegidos.append(dict(fold=f, beta_subtema=beta))
                m = evaluar(indice(cfg, corpus, entrenar_enrutador(corpus, dtr))[0], dte)
                filas.append(dict(fold=f, variante=v, n=len(dte), **m[[f"recall@{k}" for k in (1, 3, 5, 10)] + ["mrr@10", "ndcg@5"]].mean().to_dict()))
                continue
            if v == "V2":  # selección de hiperparámetros SOLO con las particiones de entrenamiento
                mejor = max(product(*GRID_V2.values()), key=lambda p: res(v, {**cfg, **dict(zip(GRID_V2, p))}, dtr)[PRINCIPAL].mean())
                cfg.update(dict(zip(GRID_V2, mejor))); elegidos.append(dict(fold=f, **dict(zip(GRID_V2, mejor))))
            m = res(v, cfg, dte)
            filas.append(dict(fold=f, variante=v, n=len(dte), **m[[f"recall@{k}" for k in (1, 3, 5, 10)] + ["mrr@10", "ndcg@5"]].mean().to_dict()))
    final_v2 = dict(zip(GRID_V2, max(product(*GRID_V2.values()), key=lambda p: res("V2", {**VARIANTES["V2"], **dict(zip(GRID_V2, p))}, dev)[PRINCIPAL].mean())))
    final_v3 = elegir_beta(corpus, dev)
    return pd.DataFrame(filas), dict(por_fold=elegidos, final=final_v2, final_v3=final_v3)


def main(ruta_bateria):
    corpus, q, fuente = cargar(ruta_bateria)
    dev, val = q[q.particion == "desarrollo"].reset_index(drop=True), q[q.particion == "validacion"].reset_index(drop=True)
    print(f"Batería {fuente}: desarrollo {len(dev)} · validación {len(val)} · evaluación final {int((q.particion == 'evaluacion_final').sum())} (sin usar)")
    commit = subprocess.run(["git", "-C", str(RAIZ), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    fecha = dt.datetime.now(dt.timezone(dt.timedelta(hours=-5))).isoformat(timespec="seconds")

    # ---------- 1) validación cruzada en desarrollo ----------
    cvdf, hp = validacion_cruzada(corpus, dev)
    VARIANTES["V2"].update(hp["final"]); VARIANTES["V3"]["beta_subtema"] = hp["final_v3"]
    enr_final = entrenar_enrutador(corpus, dev)   # entrenado con TODO desarrollo; nunca ve validación

    # ---------- 2) holdout: validación ----------
    resultados, tiempos = {}, {}
    for v, cfg in VARIANTES.items():
        conf = {"experimento": "feature-engineering-recuperacion", "variante": v, **cfg, "k_max": K_MAX, "semilla": SEMILLA,
                "corpus": "corpus_v5_1", "bateria": "bateria_v4", "particion_evaluada": "validacion"}
        texto_conf = yaml.safe_dump(conf, allow_unicode=True, sort_keys=False)
        chash = hashlib.sha256(texto_conf.encode()).hexdigest()[:12]
        run_id = f"fe-{v}-{chash}"
        carpeta = SALIDA / "runs" / run_id; carpeta.mkdir(parents=True, exist_ok=True)
        (carpeta / "config.yaml").write_text(texto_conf, encoding="utf-8")
        construcciones = [indice(cfg, corpus, enr_final if v == "V3" else None) for _ in range(5)]
        bus, t_indice = construcciones[0][0], float(np.median([t for _, t in construcciones]))
        log = open(carpeta / "log.jsonl", "w", encoding="utf-8")
        def registrar(r, top, m, lat):
            log.write(json.dumps(dict(
                run_id=run_id, config_hash=chash, commit=commit, fecha=fecha, consulta_id=r.id,
                consulta_texto=r.enunciado if r.texto_publicable else None,
                topk=[dict(chunk_id=d, score=round(s, 4), rango=i) for i, (d, s) in enumerate(top, 1)],
                relevantes=list(r.unidades_relevantes), recall_5=m["recall@5"], mrr_10=round(m["mrr@10"], 4),
                latencia_ms=round(lat, 3)), ensure_ascii=False) + "\n")
        res = evaluar(bus, val, registrar); log.close()
        resultados[v] = res
        tiempos[v] = dict(indice_ms=round(t_indice, 1), consulta_ms_media=round(res.latencia_ms.mean(), 3),
                          consulta_ms_p95=round(res.latencia_ms.quantile(0.95), 3))
        resumen = {k: float(res[k].mean()) for k in res.columns if k.startswith(("recall", "precision", "mrr", "ndcg", "fallo"))}
        (carpeta / "metricas.json").write_text(json.dumps(dict(run_id=run_id, n=len(res), **resumen, **tiempos[v]), indent=1), encoding="utf-8")
        print(f"{v}: Recall@5 = {resumen['recall@5']:.3f} · MRR@10 = {resumen['mrr@10']:.3f} · run {run_id}")

    # ---------- 3) tablas ----------
    R = SALIDA / "resultados"; R.mkdir(exist_ok=True)
    filas = []
    for v, res in resultados.items():
        lo, hi = ic_bootstrap(res[PRINCIPAL])
        fila = dict(variante=VARIANTES[v]["nombre"], cambio=VARIANTES[v]["cambio"],
                    **{"Recall@5": res["recall@5"].mean(), "IC95 Recall@5": f"[{lo:.3f}; {hi:.3f}]",
                       "Recall@1": res["recall@1"].mean(), "Recall@3": res["recall@3"].mean(), "Recall@10": res["recall@10"].mean(),
                       "MRR@10": res["mrr@10"].mean(), "nDCG@5": res["ndcg@5"].mean(), "Precision@5": res["precision@5"].mean(),
                       "Fallo@5": res["fallo@5"].mean(), "Acierto de subtema@1": res["acierto_subtema@1"].mean()}, **tiempos[v])
        if v != "B0":
            d = diferencia_pareada(resultados["B0"][PRINCIPAL], res[PRINCIPAL])
            fila.update({"Δ Recall@5 vs B0": d["diferencia"], "IC95 Δ vs B0": f"[{d['ic95'][0]:+.3f}; {d['ic95'][1]:+.3f}]", "p vs B0": d["p_bootstrap"]})
            if VARIANTES[v]["base"] != "B0":
                e = diferencia_pareada(resultados[VARIANTES[v]["base"]][PRINCIPAL], res[PRINCIPAL])
                fila.update({"Δ vs su base": e["diferencia"], "IC95 Δ vs su base": f"[{e['ic95'][0]:+.3f}; {e['ic95'][1]:+.3f}]", "p vs su base": e["p_bootstrap"]})
        filas.append(fila)
    tabla = pd.DataFrame(filas); tabla.to_csv(R / "tabla_principal.csv", index=False)
    d21 = {v: diferencia_pareada(resultados["V1"][PRINCIPAL], resultados[v][PRINCIPAL]) for v in ("V2", "V3")}
    curva = pd.DataFrame({v: [r[f"recall@{k}"].mean() for k in range(1, K_MAX + 1)] for v, r in resultados.items()}, index=range(1, K_MAX + 1))
    curva.index.name = "k"; curva.to_csv(R / "recall_vs_k.csv")
    cvres = cvdf.groupby("variante")[["recall@5", "mrr@10"]].agg(["mean", "std"]).round(3); cvdf.to_csv(R / "cv_desarrollo.csv", index=False)
    porsub = pd.DataFrame({v: r.groupby("subtema")[PRINCIPAL].mean() for v, r in resultados.items()}).round(3); porsub.to_csv(R / "recall5_por_subtema.csv")
    pornivel = pd.DataFrame({v: r.groupby(["componente", "nivel"])[PRINCIPAL].mean() for v, r in resultados.items()}).round(3); pornivel.to_csv(R / "recall5_por_nivel.csv")
    porcob = pd.DataFrame({v: r.groupby("cobertura_parcial")[PRINCIPAL].agg(["mean", "count"]).stack() for v, r in resultados.items()}).round(3); porcob.to_csv(R / "recall5_cobertura.csv")
    publicas = pd.DataFrame({v: [r.loc[r.origen.isin(["Ágora", "generada"]), PRINCIPAL].mean()] for v, r in resultados.items()}, index=["Recall@5 (solo preguntas con texto público)"]).round(3)
    publicas.to_csv(R / "recall5_solo_publicas.csv")
    json.dump(dict(fecha=fecha, commit=commit, bateria=fuente, n_desarrollo=len(dev), n_validacion=len(val),
                   hiperparametros=hp, dif_vs_v1=d21, cv=json.loads(cvres.to_json())),
              open(R / "resumen.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(tabla.round(3).to_string()); print(cvres); print("vs V1:", d21); print("Hiperparámetros V2:", hp)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--bateria", type=Path, default=None)
    main(ap.parse_args().bateria)
