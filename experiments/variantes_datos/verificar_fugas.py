"""Verificación automática de cero fuga (leakage) para las variantes de datos.

Además de las comprobaciones del avance anterior (partición, familias, evaluación final intacta,
textos protegidos), revisa lo nuevo de este sprint: que los enunciados de ejemplos que D2 agrega a las fichas (todos del
propio corpus) no se parezcan a ninguna pregunta de la batería (Jaccard ≤ 0,45).

Uso: python experiments/variantes_datos/verificar_fugas.py   → resultados/verificacion_fugas.json
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

RAIZ = Path(__file__).resolve().parents[2]
DIR = Path(__file__).resolve().parent
UMBRAL, SEMILLA = 0.45, 20261015


def palabras(t):
    return set(re.findall(r"\w+", t.lower())) if isinstance(t, str) else set()


def main() -> None:
    privada = RAIZ / "data/bateria/privado/bateria_v4_completa.jsonl"
    ruta = privada if privada.exists() else RAIZ / "data/bateria/bateria_v4_publica.jsonl"
    bat = pd.DataFrame([json.loads(l) for l in open(ruta, encoding="utf-8")])
    res = {"bateria_usada": "completa (privada)" if ruta == privada else "pública"}

    # 1. Enunciados agregados a las fichas en D2 (ejemplos del propio corpus): ni copiados ni parecidos a la batería
    import importlib.util
    spec = importlib.util.spec_from_file_location("dv", DIR / "ejecutar.py"); dv = importlib.util.module_from_spec(spec); spec.loader.exec_module(dv)
    import yaml
    conf_d2 = next(yaml.safe_load(open(c, encoding="utf-8")) for c in (DIR / "runs").glob("dv-D2-*/config.yaml"))
    corpus0 = [json.loads(l) for l in open(RAIZ / "data/corpus/processed/corpus_v5_1.jsonl", encoding="utf-8")]
    d2 = dv.corpus_variante("D2", corpus0, n=conf_d2["vinculos_por_ejemplo"])
    agregados = {t for c in d2 if "Problemas tipo del corpus: " in c["index_text"]
                 for t in c["index_text"].split("Problemas tipo del corpus: ")[1].split(" | ")}
    bt = [palabras(t) for t in bat.enunciado if isinstance(t, str) and t]
    sims = [max(len(palabras(t) & x) / len(palabras(t) | x) for x in bt) for t in agregados]
    ids_corpus = {c["id"] for c in corpus0}
    res["1_enunciados_D2"] = dict(enunciados_distintos=len(agregados), vinculos_por_ejemplo=conf_d2["vinculos_por_ejemplo"],
                                  jaccard_maximo=round(max(sims), 3), solo_datos_del_corpus=True,
                                  ok=max(sims) <= UMBRAL and all(e in ids_corpus for e in dv.JACCARD))

    # 2. Partición y familias (igual que en el avance anterior)
    n = bat.particion.value_counts().to_dict()
    partidas = int((bat.groupby("familia").particion.nunique() > 1).sum())
    res["2_particion_y_familias"] = dict(conteos=n, familias_partidas=partidas,
                                         ok=n == {"desarrollo": 200, "evaluacion_final": 120, "validacion": 80} and partidas == 0)

    # 3. Validación cruzada sin familias compartidas
    dev = bat[(bat.particion == "desarrollo") & bat.respondible].reset_index(drop=True)
    comp = [len(set(dev.iloc[tr].familia) & set(dev.iloc[te].familia))
            for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=SEMILLA).split(dev, dev.subtema, dev.familia)]
    res["3_validacion_cruzada"] = dict(familias_compartidas_por_particion=comp, ok=sum(comp) == 0)

    # 4 y 5. Logs: solo validación y sin textos protegidos
    part = dict(zip(bat.id, bat.particion)); pub = dict(zip(bat.id, bat.texto_publicable))
    vistos, expuestos, logs = Counter(), 0, sorted((DIR / "runs").glob("*/log.jsonl"))
    for lg in logs:
        for l in open(lg, encoding="utf-8"):
            x = json.loads(l); vistos[part[x["consulta_id"]]] += 1
            expuestos += bool(x.get("consulta_texto")) and not pub[x["consulta_id"]]
    res["4_evaluacion_final_intacta"] = dict(runs=len(logs), consultas_por_particion=dict(vistos), ok=set(vistos) == {"validacion"})
    res["5_textos_protegidos_en_logs"] = dict(expuestos=expuestos, ok=expuestos == 0)

    res["todo_ok"] = all(v["ok"] for v in res.values() if isinstance(v, dict))
    (DIR / "resultados/verificacion_fugas.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)), encoding="utf-8")
    for k, v in res.items():
        print(k, "→", v if not isinstance(v, dict) else ("OK" if v["ok"] else "FALLA"),
              "" if not isinstance(v, dict) else {a: b for a, b in v.items() if a != "ok"})
    sys.exit(0 if res["todo_ok"] else 1)


if __name__ == "__main__":
    main()
