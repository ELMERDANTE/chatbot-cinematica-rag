"""Verificación automática de cero fuga (leakage) y de la partición (semana 5).

Comprueba, sobre los datos y sobre los logs de los runs:
  1. Ninguna pregunta de la batería está copiada en el corpus (similitud de Jaccard).
  2. La partición es 200/80/120 y cada estrato subtema × componente × nivel se reparte 50/20/30 %.
  3. Ninguna familia de preguntas parecidas queda partida entre conjuntos.
  4. Las particiones de la validación cruzada no parten familias y conservan la proporción de subtemas.
  5. Los logs de los runs solo contienen preguntas de validación (la evaluación final no se tocó).
  6. Los logs no exponen enunciados de libros protegidos.

Uso: python experiments/semana5/verificar_fugas.py   → resultados/verificacion_fugas.json
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "src"))
from rag.preprocesamiento import tokens_basico  # noqa: E402

SEMILLA = 20261015
DIR = Path(__file__).resolve().parent


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a | b else 0.0


def main() -> None:
    corpus = [json.loads(l) for l in open(RAIZ / "data/corpus/processed/corpus_v5_1.jsonl", encoding="utf-8")]
    privada = RAIZ / "data/bateria/privado/bateria_v4_completa.jsonl"
    ruta = privada if privada.exists() else RAIZ / "data/bateria/bateria_v4_publica.jsonl"
    bat = pd.DataFrame([json.loads(l) for l in open(ruta, encoding="utf-8")])
    res: dict = {"bateria_usada": "completa (privada)" if ruta == privada else "pública"}

    # 1. Copias de preguntas en el corpus
    fichas = [(c["id"], set(tokens_basico(c["index_text"]))) for c in corpus]
    maximos = []
    for r in bat[bat.enunciado.notna() & (bat.enunciado != "")].itertuples():
        t = set(tokens_basico(r.enunciado))
        maximos.append(max(jaccard(t, f) for _, f in fichas))
    res["1_copias_en_corpus"] = dict(preguntas_revisadas=len(maximos), jaccard_maximo=round(max(maximos), 3),
                                     preguntas_con_jaccard_mayor_0_5=sum(m > 0.5 for m in maximos),
                                     ok=max(maximos) <= 0.5)

    # 2. Partición y estratos
    n = bat.particion.value_counts().to_dict()
    objetivo = pd.Series({"desarrollo": .5, "validacion": .2, "evaluacion_final": .3})
    bat["estrato"] = bat.componente.where(bat.componente != "practica", "practica·" + bat.nivel.astype(str))
    desv = {}
    for nombre, col in [("estrato (teoría / práctica×nivel / no respondible)", bat.estrato), ("subtema", bat.subtema)]:
        prop = pd.crosstab(col, bat.particion, normalize="index")
        desv[nombre] = round(float((prop - objetivo).abs().max().max()), 4)
    # Informativo: dentro de cada subtema×práctica×nivel (18 preguntas) el reparto exacto 9/3,6/5,4 es imposible.
    fino = pd.crosstab(bat.subtema + "|" + bat.estrato, bat.particion)
    res["2_particion"] = dict(conteos=n, desviacion_maxima=desv,
                              practica_subtema_nivel_desarrollo_siempre_9=bool((fino.filter(like="practica", axis=0).desarrollo == 9).all()),
                              ok=n == {"desarrollo": 200, "evaluacion_final": 120, "validacion": 80} and max(desv.values()) < 1e-9)

    # 3. Familias
    part_por_fam = bat.groupby("familia").particion.nunique()
    res["3_familias"] = dict(familias=int(len(part_por_fam)), familias_con_varias_preguntas=int((bat.familia.value_counts() > 1).sum()),
                             familias_partidas=int((part_por_fam > 1).sum()), ok=int((part_por_fam > 1).sum()) == 0)

    # 4. Particiones de la validación cruzada (mismo generador y semilla que ejecutar.py)
    dev = bat[(bat.particion == "desarrollo") & bat.respondible].reset_index(drop=True)
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEMILLA)
    folds = []
    for f, (tr, te) in enumerate(cv.split(dev, dev.subtema, dev.familia), 1):
        dtr, dte = dev.iloc[tr], dev.iloc[te]
        dist = dte.subtema.value_counts(normalize=True)
        folds.append(dict(fold=f, n_prueba=len(dte), familias_compartidas=len(set(dtr.familia) & set(dte.familia)),
                          proporcion_subtema_min=round(float(dist.min()), 3), proporcion_subtema_max=round(float(dist.max()), 3)))
    res["4_validacion_cruzada"] = dict(folds=folds, ok=all(x["familias_compartidas"] == 0 for x in folds))

    # 5 y 6. Logs de los runs
    part = dict(zip(bat.id, bat.particion))
    publicable = dict(zip(bat.id, bat.get("texto_publicable", pd.Series([True] * len(bat)))))
    logs = sorted((DIR / "runs").glob("*/log.jsonl"))
    vistos, expuestos = Counter(), 0
    for lg in logs:
        for l in open(lg, encoding="utf-8"):
            x = json.loads(l)
            vistos[part[x["consulta_id"]]] += 1
            expuestos += bool(x.get("consulta_texto")) and not publicable[x["consulta_id"]]
    res["5_evaluacion_final_intacta"] = dict(runs=len(logs), consultas_por_particion=dict(vistos),
                                            ok=set(vistos) == {"validacion"})
    res["6_textos_protegidos_en_logs"] = dict(expuestos=expuestos, ok=expuestos == 0)

    res["todo_ok"] = all(v["ok"] for k, v in res.items() if isinstance(v, dict) and "ok" in v)
    (DIR / "resultados/verificacion_fugas.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)), encoding="utf-8")
    for k, v in res.items():
        print(k, "→", v if not isinstance(v, dict) else ("OK" if v.get("ok") else "FALLA"), "" if not isinstance(v, dict) else {a: b for a, b in v.items() if a not in ("ok", "folds")})
    sys.exit(0 if res["todo_ok"] else 1)


if __name__ == "__main__":
    main()
