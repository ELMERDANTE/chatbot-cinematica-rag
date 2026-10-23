"""Verificación automática de cero fuga (leakage) para las variantes de datos.

Además de las comprobaciones del avance anterior (partición, familias, evaluación final intacta,
textos protegidos), revisa lo nuevo de este sprint: que los problemas agregados al corpus (D2)
no sean preguntas de la batería ni se parezcan a ninguna (Jaccard ≤ 0,45).

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
    exp = [json.loads(l) for l in open(RAIZ / "data/corpus/variantes/expansion_agora.jsonl", encoding="utf-8")]
    problemas = {p["fuente"]: p["enunciado"] for e in exp for p in e["problemas"]}
    res = {"bateria_usada": "completa (privada)" if ruta == privada else "pública"}

    # 1. Problemas agregados (D2): ni copiados ni parecidos a la batería
    bt = [palabras(t) for t in bat.enunciado if isinstance(t, str) and t]
    sims = {f: max(len(palabras(t) & x) / len(palabras(t) | x) for x in bt) for f, t in problemas.items()}
    en_bateria = set(problemas) & set(bat.fuente)
    res["1_expansion_D2"] = dict(problemas_distintos=len(problemas), presentes_en_bateria=len(en_bateria),
                                 jaccard_maximo=round(max(sims.values()), 3), ok=not en_bateria and max(sims.values()) <= UMBRAL)

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
