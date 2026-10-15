"""Genera los datos públicos del repositorio a partir de los archivos de trabajo (privados).

- Corpus: se publica completo (síntesis propias y problemas originales).
- Batería: texto completo solo para preguntas de la academia Ágora y generadas;
  para las de libros con derechos de autor (Lumbreras, material escolar) se publica
  la referencia exacta, la clave, la solución de referencia y los metadatos.

Uso: python src/preparar_datos_publicos.py --privado <carpeta_privada>
"""
import argparse, csv, hashlib, json, re, collections
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
def palabras(t): return len(re.findall(r"\w+", t or ""))

def main(priv):
    priv = Path(priv)
    # ---------------- corpus ----------------
    src = priv / "corpus_v5.jsonl"
    filas = [json.loads(l) for l in open(src, encoding="utf-8")]
    out = RAIZ / "data/corpus/processed"; out.mkdir(parents=True, exist_ok=True)
    with open(out / "corpus_v5_1.jsonl", "w", encoding="utf-8") as f:
        for r in filas: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (out / "corpus_v5_1_lectura.md").write_text((priv / "corpus_v5_lectura.md").read_text(encoding="utf-8"), encoding="utf-8")
    manifest = dict(version="v5.1", n_fichas=len(filas),
                    sha256_archivo=hashlib.sha256((out / "corpus_v5_1.jsonl").read_bytes()).hexdigest(),
                    fichas=[dict(id=r["id"], subtema=r["subtema_proyecto"], tipo=r["tipo"], nivel=r["nivel"], titulo=r["titulo"],
                                 sha256=hashlib.sha256(json.dumps(r, ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:16])
                            for r in filas])
    (RAIZ / "data/corpus/manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    # ---------------- batería ----------------
    B = [json.loads(l) for l in open(priv / "bateria_v4.jsonl", encoding="utf-8")]
    bd = RAIZ / "data/bateria"; bd.mkdir(parents=True, exist_ok=True)
    with open(bd / "bateria_v4_publica.jsonl", "w", encoding="utf-8") as f:
        for b in B:
            p = dict(b)
            if not b["texto_publicable"]:
                p["enunciado"] = None; p["alternativas"] = None
                p["nota_publicacion"] = "Texto no publicado por derechos de autor; ver la referencia exacta en «fuente»."
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    campos = ["id","subtema","componente","nivel","origen","fuente","familia","particion","clave","respondible","g",
              "verificacion","texto_publicable","modificada","cobertura_parcial","n_unidades","unidades_relevantes",
              "n_palabras_enunciado","n_caracteres_enunciado","n_alternativas","n_similares_corpus"]
    with open(bd / "bateria_v4_metadatos.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=campos); w.writeheader()
        for b in B:
            w.writerow(dict(id=b["id"], subtema=b["subtema"], componente=b["componente"], nivel=b["nivel"], origen=b["origen"],
                fuente=b["fuente"], familia=b["familia"], particion=b["particion"], clave=b["clave"] or "", respondible=b["respondible"],
                g=b["g"], verificacion=b["verificacion"], texto_publicable=b["texto_publicable"], modificada=bool(b["modificacion"]),
                cobertura_parcial=b.get("cobertura_parcial", False), n_unidades=len(b["unidades_relevantes"]),
                unidades_relevantes=";".join(b["unidades_relevantes"]), n_palabras_enunciado=palabras(b["enunciado"]),
                n_caracteres_enunciado=len(b["enunciado"]), n_alternativas=len(b["alternativas"] or {}),
                n_similares_corpus=len(b.get("similar_a_corpus", []))))
    # registro de verificación y decisiones
    reg = []
    for b in B:
        if b["modificacion"]: reg.append(dict(id=b["id"], fuente=b["fuente"], decision="modificada", detalle=b["modificacion"]))
    for l in open(priv / "claves_docente.tsv", encoding="utf-8"):
        p = l.rstrip("\n").split("\t")
        if len(p) > 4 and re.search(r"ANULADA|OMITIDA|EXCLUIDA", p[4]):
            reg.append(dict(id="", fuente=f"{p[0]} {p[1]} n.º {p[2]}", decision=p[4].split(":")[0].lower(), detalle=p[4]))
    with open(bd / "registro_verificacion.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id","fuente","decision","detalle"]); w.writeheader(); w.writerows(reg)
    # inventario de fuentes (embudo)
    inv = json.load(open(priv / "inventario_fuentes.json", encoding="utf-8"))
    ed = RAIZ / "data/eda"; ed.mkdir(parents=True, exist_ok=True)
    with open(ed / "inventario_fuentes.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(inv[0].keys())); w.writeheader(); w.writerows(inv)
    print(f"corpus: {len(filas)} fichas · batería: {len(B)} preguntas · registro: {len(reg)} decisiones")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--privado", required=True); main(ap.parse_args().privado)
