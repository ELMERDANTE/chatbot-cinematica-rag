"""Calcula, para cada ejemplo resuelto del corpus v5.1, su parecido máximo con la batería (variante de datos D2).

D2 usa SOLO datos del propio corpus: el enunciado de cada ejemplo resuelto se agrega al texto indexado de
las fichas de teoría/procedimiento de su mismo subtema con las que se vincula (ver
experiments/variantes_datos/ejecutar.py). Para no trasladar a las fichas enunciados parecidos al examen, se
aplica la misma regla anti-fuga del proyecto: se descarta todo ejemplo cuya similitud de Jaccard con
CUALQUIER pregunta de la batería (desarrollo, validación o evaluación final) supere 0,45.

Este script necesita la batería completa (privada) y publica solo números: el id de cada ejemplo y su
Jaccard máximo. Salida: data/corpus/variantes/ejemplos_jaccard.json

Uso: python src/preparar_ejemplos_d2.py --privado <carpeta_privada>
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]


def palabras(t: str | None) -> set[str]:
    return set(re.findall(r"\w+", (t or "").lower()))


def main(priv: Path) -> None:
    bateria = [json.loads(l) for l in open(priv / "bateria_v4.jsonl", encoding="utf-8")]
    bt = [palabras(b["enunciado"]) for b in bateria if b["enunciado"]]
    corpus = [json.loads(l) for l in open(RAIZ / "data/corpus/processed/corpus_v5_1.jsonl", encoding="utf-8")]
    sal = {}
    for c in corpus:
        if c.get("tipo") != "ejemplo":
            continue
        t = palabras(c["problema"])
        sal[c["id"]] = round(max(len(t & x) / len(t | x) for x in bt), 3)
    out = RAIZ / "data/corpus/variantes"; out.mkdir(parents=True, exist_ok=True)
    (out / "ejemplos_jaccard.json").write_text(json.dumps(dict(sorted(sal.items())), indent=1), encoding="utf-8")
    print(f"{len(sal)} ejemplos · > 0,45: {sum(v > 0.45 for v in sal.values())}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--privado", required=True, type=Path)
    main(ap.parse_args().privado)
