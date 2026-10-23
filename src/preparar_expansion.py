"""Prepara la expansión del corpus con problemas reales de la academia Ágora (variante de datos Var2).

Toma la reserva de preguntas de práctica de Ágora que se verificaron pero NO entraron a la batería
(archivo privado `reserva_agora.json`) y, para cada ficha del corpus, guarda los enunciados de los
problemas de la reserva que usan esa ficha (según su anotación `unidades_relevantes`).

Control de fuga: se descarta todo problema de la reserva cuya similitud de Jaccard con CUALQUIER
pregunta de la batería (desarrollo, validación o evaluación final) supere 0,45, el mismo umbral que
usa el EDA para marcar preguntas de «estructura parecida».

Salida pública: data/corpus/variantes/expansion_agora.jsonl (una línea por ficha expandida).

Uso: python src/preparar_expansion.py --privado <carpeta_privada>
"""
from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
UMBRAL = 0.45
MAX_POR_FICHA = 5


def palabras(t: str | None) -> set[str]:
    return set(re.findall(r"\w+", (t or "").lower()))


def main(priv: Path) -> None:
    reserva = json.load(open(priv / "reserva_agora.json", encoding="utf-8"))
    bateria = [json.loads(l) for l in open(priv / "bateria_v4.jsonl", encoding="utf-8")]
    corpus_ids = {json.loads(l)["id"] for l in open(RAIZ / "data/corpus/processed/corpus_v5_1.jsonl", encoding="utf-8")}
    fuentes_bat = {b["fuente"] for b in bateria}
    bt = [palabras(b["enunciado"]) for b in bateria if b["enunciado"]]

    aceptados, excluidos = [], []
    for r in reserva:
        assert r["fuente"] not in fuentes_bat, "la reserva no debe contener preguntas de la batería"
        t = palabras(r["enunciado"])
        sim = max(len(t & x) / len(t | x) for x in bt)
        (excluidos if sim > UMBRAL else aceptados).append({**r, "jaccard_max_bateria": round(sim, 3)})

    por_ficha = collections.defaultdict(list)
    for r in sorted(aceptados, key=lambda r: r["fuente"]):
        for u in r["unidades_relevantes"]:
            assert u in corpus_ids, u
            if len(por_ficha[u]) < MAX_POR_FICHA:
                por_ficha[u].append({"fuente": r["fuente"], "enunciado": r["enunciado"], "jaccard_max_bateria": r["jaccard_max_bateria"]})

    out = RAIZ / "data/corpus/variantes"; out.mkdir(parents=True, exist_ok=True)
    with open(out / "expansion_agora.jsonl", "w", encoding="utf-8") as f:
        for fid in sorted(por_ficha):
            f.write(json.dumps({"ficha": fid, "problemas": por_ficha[fid]}, ensure_ascii=False) + "\n")
    resumen = dict(reserva_agora=len(reserva), excluidos_por_parecido=len(excluidos), aceptados=len(aceptados),
                   umbral_jaccard=UMBRAL, max_por_ficha=MAX_POR_FICHA, fichas_expandidas=len(por_ficha),
                   problemas_agregados=sum(len(v) for v in por_ficha.values()),
                   jaccard_max_aceptado=max(r["jaccard_max_bateria"] for r in aceptados))
    (out / "expansion_agora_resumen.json").write_text(json.dumps(resumen, ensure_ascii=False, indent=1), encoding="utf-8")
    print(resumen)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--privado", required=True, type=Path)
    main(ap.parse_args().privado)
