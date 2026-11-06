"""Verificación automática de cero fuga para la optimización de hiperparámetros.

Comprueba que (1) las particiones de la CV no comparten familias, (2) los registros de las ejecuciones
contienen solo preguntas de validación, (3) la evaluación final no se usó y (4) no se publican textos
protegidos. La selección de hiperparámetros usa solo desarrollo por construcción (ver ejecutar.py).

Uso: python experiments/optimizacion_hiperparametros/verificar_fugas.py → resultados/verificacion_fugas.json
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

RAIZ = Path(__file__).resolve().parents[2]
DIR = Path(__file__).resolve().parent


def main() -> None:
    privada = RAIZ / "data/bateria/privado/bateria_v4_completa.jsonl"
    ruta = privada if privada.exists() else RAIZ / "data/bateria/bateria_v4_publica.jsonl"
    bat = pd.DataFrame([json.loads(l) for l in open(ruta, encoding="utf-8")])
    res = {"bateria_usada": "completa (privada)" if ruta == privada else "pública"}
    dev = bat[(bat.particion == "desarrollo") & bat.respondible].reset_index(drop=True)
    comp = [len(set(dev.iloc[tr].familia) & set(dev.iloc[te].familia))
            for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=20261015).split(dev, dev.subtema, dev.familia)]
    res["1_cv_sin_familias_compartidas"] = dict(familias_compartidas=comp, ok=sum(comp) == 0)
    part = bat.set_index("id").particion
    logs = [json.loads(l) for f in (DIR / "runs").glob("*/log.jsonl") for l in open(f, encoding="utf-8")]
    cuenta = pd.Series([part[x["consulta_id"]] for x in logs]).value_counts().to_dict()
    res["2_logs_solo_validacion"] = dict(consultas_por_particion=cuenta, ok=set(cuenta) == {"validacion"})
    resumen = json.load(open(DIR / "resultados/resumen.json", encoding="utf-8"))
    res["3_evaluacion_final_intacta"] = dict(ok=not resumen["evaluacion_final_usada"] and "evaluacion_final" not in cuenta)
    publ = bat.set_index("id").texto_publicable
    expuestos = sum(1 for x in logs if x["consulta_texto"] and not publ[x["consulta_id"]])
    res["4_textos_protegidos_en_logs"] = dict(expuestos=expuestos, ok=expuestos == 0)
    res["todo_ok"] = all(v["ok"] for k, v in res.items() if isinstance(v, dict))
    (DIR / "resultados/verificacion_fugas.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    for k, v in res.items():
        print(k, "→", ("OK" if v.get("ok") else "FALLA") if isinstance(v, dict) else v, v if isinstance(v, dict) else "")


if __name__ == "__main__":
    main()
