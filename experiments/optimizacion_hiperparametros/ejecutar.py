"""Optimización de hiperparámetros del buscador (sprint «optimización de hiperparámetros»).

Punto de partida: la configuración vigente, el buscador V1 (BM25 + palabras vacías + raíces, k1 = 1,5,
b = 0,75, sin enrutador) sobre el corpus v5.1.

Espacio de búsqueda (rejilla completa, 576 configuraciones):
  - preprocesamiento: basico | raices
  - k1 (saturación de la frecuencia de un término en BM25): 0,5 · 0,8 · 1,2 · 1,5 · 2,0 · 2,5 · 3,0 · 4,0
  - b (normalización por longitud de la ficha en BM25): 0 · 0,1 · 0,25 · 0,5 · 0,75 · 1,0
  - peso del título (veces que se repite el título en el texto indexado): 1 · 2 · 3
  - ejemplos: completos | sin_enunciado (se indexan las fichas de ejemplo resuelto sin la línea
    «Problema:»; hallazgo del sprint de importancia de variables)

Protocolo (sin fuga):
  1. Selección con validación cruzada de 5 particiones en DESARROLLO (StratifiedGroupKFold por subtema y
     familia, semilla 20261015). Criterio: mayor Recall@5 medio; en empate, la configuración más
     parecida a la vigente.
  2. Estimación honesta del rendimiento tras ajustar: validación cruzada ANIDADA (la selección se repite
     dentro de cada partición externa usando solo las otras 4).
  3. Comparación final, una sola vez, en el holdout de VALIDACIÓN: configuración vigente vs. optimizada,
     con bootstrap pareado. Se reporta aparte la optimización solo de BM25 (ejemplos completos) para
     separar los dos efectos.
  La evaluación final no se usa. El número de fichas k no se optimiza con Recall (siempre crece con k):
  se fija en 5 por el presupuesto de contexto del LLM (EDA: ≈ 800–1200 tokens).

Uso: python experiments/optimizacion_hiperparametros/ejecutar.py
"""
from __future__ import annotations

import datetime as dt
import hashlib
import itertools
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import StratifiedGroupKFold

RAIZ = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(RAIZ / "src"), str(RAIZ / "experiments/feature_engineering")]
import ejecutar as fe  # noqa: E402
from rag.metricas import diferencia_pareada, ic_bootstrap  # noqa: E402

SEMILLA, PRINCIPAL, K_MAX = 20261015, "recall@5", 10
DIR = Path(__file__).resolve().parent
R = DIR / "resultados"

ESPACIO = {
    "preprocesamiento": ["basico", "raices"],
    "k1": [0.5, 0.8, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0],
    "b": [0.0, 0.1, 0.25, 0.5, 0.75, 1.0],
    "peso_titulo": [1, 2, 3],
    "ejemplos": ["completos", "sin_enunciado"],
}
VIGENTE = dict(preprocesamiento="raices", k1=1.5, b=0.75, peso_titulo=1, ejemplos="completos")


def preparar_corpus(corpus: list[dict], peso_titulo: int, ejemplos: str) -> list[dict]:
    salida = []
    for c in corpus:
        lineas = c["index_text"].split("\n")
        if ejemplos == "sin_enunciado" and c.get("tipo") == "ejemplo":
            lineas = [l for l in lineas if not l.startswith("Problema:")]
        lineas = [lineas[0]] * peso_titulo + lineas[1:]
        salida.append({**c, "index_text": "\n".join(lineas)})
    return salida


def construir(cfg: dict, corpus: list[dict]):
    c = preparar_corpus(corpus, cfg["peso_titulo"], cfg["ejemplos"])
    return fe.indice({k: cfg[k] for k in ("preprocesamiento", "k1", "b")}, c)[0]


def distancia(cfg: dict) -> int:
    """N.º de hiperparámetros que difieren de la configuración vigente (desempate: lo más simple)."""
    return sum(cfg[k] != VIGENTE[k] for k in VIGENTE)


def configuraciones(fijos: dict | None = None):
    claves = list(ESPACIO)
    for valores in itertools.product(*ESPACIO.values()):
        cfg = dict(zip(claves, valores))
        if fijos and any(cfg[k] != v for k, v in fijos.items()):
            continue
        yield cfg


def elegir(tabla: pd.DataFrame, columnas: list[str]) -> dict:
    """Mejor configuración según la media de Recall@5 en las particiones dadas; desempate por simplicidad."""
    t = tabla.assign(media=tabla[columnas].mean(axis=1), dist=tabla.apply(lambda r: distancia(r), axis=1))
    mejor = t.sort_values(["media", "dist"], ascending=[False, True]).iloc[0]
    return {k: (mejor[k].item() if hasattr(mejor[k], "item") else mejor[k]) for k in ESPACIO}


def main():
    corpus, q, fuente = fe.cargar(None)
    dev = q[q.particion == "desarrollo"].reset_index(drop=True)
    val = q[q.particion == "validacion"].reset_index(drop=True)
    R.mkdir(parents=True, exist_ok=True)
    commit = subprocess.run(["git", "-C", str(RAIZ), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    fecha = dt.datetime.now(dt.timezone(dt.timedelta(hours=-5))).isoformat(timespec="seconds")
    print(f"Batería {fuente}: desarrollo {len(dev)} · validación {len(val)}")

    folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=SEMILLA).split(dev, dev.subtema, dev.familia))
    for tr, te in folds:
        assert not set(dev.iloc[tr].familia) & set(dev.iloc[te].familia)

    # 1. Rejilla completa evaluada SOLO en desarrollo (Recall@5 por partición)
    t0 = time.time(); filas = []
    for cfg in configuraciones():
        r = fe.evaluar(construir(cfg, corpus), dev).set_index("id")
        filas.append({**cfg, **{f"fold{i}": float(r.loc[dev.iloc[te].id, PRINCIPAL].mean()) for i, (_, te) in enumerate(folds, 1)}})
    rejilla = pd.DataFrame(filas)
    F = [f"fold{i}" for i in range(1, 6)]
    rejilla["recall5_cv"] = rejilla[F].mean(axis=1); rejilla["de_cv"] = rejilla[F].std(axis=1)
    rejilla["cambios_vs_vigente"] = rejilla.apply(distancia, axis=1)
    rejilla.sort_values("recall5_cv", ascending=False).round(4).to_csv(R / "rejilla_cv.csv", index=False)
    print(f"Rejilla: {len(rejilla)} configuraciones en {time.time() - t0:.0f} s")

    optima = elegir(rejilla, F)
    # Control de borde: si el k1 óptimo es el mayor de la rejilla, se comprueba que más allá no mejore (meseta)
    if optima["k1"] == max(ESPACIO["k1"]):
        borde = []
        for k1 in (optima["k1"], 6.0, 8.0, 12.0):
            r = fe.evaluar(construir({**optima, "k1": k1}, corpus), dev).set_index("id")
            borde.append(dict(k1=k1, recall5_cv=round(float(np.mean([r.loc[dev.iloc[te].id, PRINCIPAL].mean() for _, te in folds])), 4)))
        pd.DataFrame(borde).to_csv(R / "control_borde_k1.csv", index=False); print("Control de borde k1:", borde)
    solo_bm25 = elegir(rejilla[rejilla.ejemplos == "completos"], F)
    print("Óptima:", optima, "| solo BM25:", solo_bm25)

    # 2. CV anidada: estimación sin optimismo de lo que se gana al ajustar
    anidada = []
    for i, nombre in enumerate(F):
        internas = [f for f in F if f != nombre]
        sel = elegir(rejilla, internas)
        fila_sel = rejilla.loc[(rejilla[list(ESPACIO)] == pd.Series(sel)).all(axis=1)].iloc[0]
        fila_vig = rejilla.loc[(rejilla[list(ESPACIO)] == pd.Series(VIGENTE)).all(axis=1)].iloc[0]
        anidada.append(dict(particion_externa=i + 1, **{f"sel_{k}": v for k, v in sel.items()},
                            recall5_vigente=fila_vig[nombre], recall5_seleccionada=fila_sel[nombre]))
    anidada = pd.DataFrame(anidada); anidada.round(4).to_csv(R / "cv_anidada.csv", index=False)
    print(anidada.round(3).to_string(index=False))

    # 3. Importancia de los hiperparámetros: efecto marginal (media de Recall@5 CV por valor)
    marg = []
    for k, vals in ESPACIO.items():
        medias = rejilla.groupby(k)["recall5_cv"].agg(["mean", "max"])
        for v in vals:
            marg.append(dict(hiperparametro=k, valor=v, recall5_cv_medio=medias.loc[v, "mean"], recall5_cv_mejor=medias.loc[v, "max"]))
    marg = pd.DataFrame(marg); marg.round(4).to_csv(R / "efecto_marginal.csv", index=False)
    rango = marg.groupby("hiperparametro").recall5_cv_medio.agg(lambda s: s.max() - s.min()).sort_values(ascending=False)
    rango.rename("rango_del_recall5_medio").round(4).to_csv(R / "importancia_hiperparametros.csv")
    # Mapa k1 × b con el resto de hiperparámetros en su valor óptimo
    fijo = {k: optima[k] for k in ("preprocesamiento", "peso_titulo", "ejemplos")}
    mapa = rejilla[(rejilla[list(fijo)] == pd.Series(fijo)).all(axis=1)].pivot(index="k1", columns="b", values="recall5_cv")
    mapa.round(4).to_csv(R / "mapa_k1_b.csv")

    # 4. Comparación final en validación (una vez), con registro de cada ejecución
    candidatas = {"vigente": VIGENTE, "solo_bm25": solo_bm25, "optimizada": optima}
    res = {}
    for nombre, cfg in candidatas.items():
        conf = {"experimento": "optimizacion-hiperparametros", "candidata": nombre, **cfg, "enrutador": False,
                "corpus": "v5.1", "k_contexto": 5, "k_max": K_MAX, "semilla": SEMILLA, "bateria": "bateria_v4",
                "particion_evaluada": "validacion", "seleccion": "CV de 5 particiones en desarrollo"}
        texto = yaml.safe_dump(conf, allow_unicode=True, sort_keys=False)
        chash = hashlib.sha256(texto.encode()).hexdigest()[:12]; run_id = f"ho-{nombre}-{chash}"
        carpeta = DIR / "runs" / run_id; carpeta.mkdir(parents=True, exist_ok=True)
        (carpeta / "config.yaml").write_text(texto, encoding="utf-8")
        log = open(carpeta / "log.jsonl", "w", encoding="utf-8")

        def registrar(r, top, m, lat):
            log.write(json.dumps(dict(run_id=run_id, config_hash=chash, commit=commit, fecha=fecha, consulta_id=r.id,
                                      consulta_texto=r.enunciado if r.texto_publicable else None,
                                      topk=[dict(chunk_id=d, score=round(s, 4), rango=i) for i, (d, s) in enumerate(top, 1)],
                                      relevantes=list(r.unidades_relevantes), recall_5=m["recall@5"], mrr_10=round(m["mrr@10"], 4),
                                      latencia_ms=round(lat, 3)), ensure_ascii=False) + "\n")
        r = fe.evaluar(construir(cfg, corpus), val, registrar); log.close(); res[nombre] = r
        resumen = {k: float(r[k].mean()) for k in r.columns if k.startswith(("recall", "mrr", "ndcg", "fallo", "acierto"))}
        (carpeta / "metricas.json").write_text(json.dumps(dict(run_id=run_id, n=len(r), **resumen,
                                                               consulta_ms=round(float(r.latencia_ms.mean()), 3)), indent=1), encoding="utf-8")
        print(f"{nombre}: Recall@5 validación = {resumen['recall@5']:.3f} · run {run_id}")

    filas = []
    for nombre, r in res.items():
        lo, hi = ic_bootstrap(r[PRINCIPAL]); cfg = candidatas[nombre]
        fila_cv = rejilla.loc[(rejilla[list(ESPACIO)] == pd.Series(cfg)).all(axis=1)].iloc[0]
        f = dict(candidata=nombre, **cfg, recall5_cv=fila_cv.recall5_cv, de_cv=fila_cv.de_cv,
                 recall5_val=r[PRINCIPAL].mean(), ic95_val=f"[{lo:.3f}; {hi:.3f}]",
                 recall1=r["recall@1"].mean(), recall10=r["recall@10"].mean(), mrr10=r["mrr@10"].mean(), fallo5=r["fallo@5"].mean(),
                 practica=r.loc[r.componente == "practica", PRINCIPAL].mean(), teoria=r.loc[r.componente == "teoria", PRINCIPAL].mean(),
                 consulta_ms=r.latencia_ms.mean())
        if nombre != "vigente":
            d = diferencia_pareada(res["vigente"][PRINCIPAL], r[PRINCIPAL])
            f.update(delta_vs_vigente=d["diferencia"], ic95_delta=f"[{d['ic95'][0]:+.3f}; {d['ic95'][1]:+.3f}]", p=d["p_bootstrap"])
        filas.append(f)
    tabla = pd.DataFrame(filas); tabla.round(4).to_csv(R / "tabla_final.csv", index=False)
    curva = pd.DataFrame({n: [r[f"recall@{k}"].mean() for k in range(1, K_MAX + 1)] for n, r in res.items()}, index=range(1, K_MAX + 1))
    curva.index.name = "k"; curva.round(4).to_csv(R / "recall_vs_k.csv")
    json.dump(dict(fecha=fecha, commit=commit, bateria=fuente, n_desarrollo=len(dev), n_validacion=len(val),
                   n_configuraciones=len(rejilla), espacio=ESPACIO, vigente=VIGENTE, optima=optima, solo_bm25=solo_bm25,
                   cv_anidada=dict(vigente=round(float(anidada.recall5_vigente.mean()), 4),
                                   seleccionada=round(float(anidada.recall5_seleccionada.mean()), 4),
                                   particiones_en_que_mejora=int((anidada.recall5_seleccionada > anidada.recall5_vigente + 1e-9).sum())),
                   importancia=rango.round(4).to_dict(), evaluacion_final_usada=False),
              open(R / "resumen.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(tabla.round(3).to_string(index=False)); print(rango.round(3))


if __name__ == "__main__":
    main()
