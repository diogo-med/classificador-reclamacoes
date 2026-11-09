"""Sorteia erros do modelo ajustado no teste para análise qualitativa (etapa 11).

"Erro" = conjunto de categorias ou sentimento diferente do rótulo de referência (hoje, o Gemini;
depois da revisão, o humano). O sorteio é fixo (SEED) para a análise ser reproduzível.
Grava resultados/analise_erros.json; os comentários ficam em resultados/analise_erros.md,
escritos à mão a partir desse arquivo.

Uso: python src/analise_erros.py [--n 10] [--pred resultados/predicoes/qwen2b_lora_teste.jsonl]
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATA_PROCESSED, RESULTADOS, SEED  # noqa: E402
from esquema import avaliar_resposta  # noqa: E402


def referencia() -> tuple[pd.DataFrame, str]:
    revisados = DATA_PROCESSED / "rotulos_revisados.parquet"
    if revisados.exists():
        return pd.read_parquet(revisados), "revisão humana"
    return pd.read_parquet(DATA_PROCESSED / "rotulos_gemini.parquet"), "Gemini (provisório)"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--pred", default=str(RESULTADOS / "predicoes" / "qwen2b_lora_teste.jsonl"))
    a = p.parse_args()

    ref, nome_ref = referencia()
    amostra = pd.read_parquet(DATA_PROCESSED / "amostra.parquet")
    teste = amostra[amostra["split"] == "teste"][["review_id", "texto", "nota", "atraso_real"]]
    linhas = []
    for l in Path(a.pred).read_text(encoding="utf-8").splitlines():
        d = json.loads(l)
        r = avaliar_resposta(d["resposta"])
        linhas.append({"review_id": d["review_id"], "pred_sentimento": r["sentimento"],
                       "pred_categorias": sorted(r["categorias"] or [])})
    pred = pd.DataFrame(linhas)
    df = (teste.merge(ref[["review_id", "sentimento", "categorias"]], on="review_id")
          .merge(pred, on="review_id"))
    df["categorias"] = df["categorias"].apply(lambda c: sorted(list(c)))
    erro = (df["categorias"] != df["pred_categorias"]) | (df["sentimento"] != df["pred_sentimento"])
    erros = df[erro]
    sorteio = erros.sample(n=min(a.n, len(erros)), random_state=SEED).sort_values("review_id")
    saida = {
        "referencia": nome_ref, "previsoes": Path(a.pred).name,
        "n_teste": int(len(df)), "n_erros": int(erro.sum()),
        "so_categorias": int(((df["categorias"] != df["pred_categorias"])
                              & (df["sentimento"] == df["pred_sentimento"])).sum()),
        "so_sentimento": int(((df["categorias"] == df["pred_categorias"])
                              & (df["sentimento"] != df["pred_sentimento"])).sum()),
        "exemplos": [
            {"review_id": r.review_id, "nota": int(r.nota),
             "atraso_real": None if pd.isna(r.atraso_real) else bool(r.atraso_real),
             "texto": r.texto, "referencia": {"sentimento": r.sentimento, "categorias": r.categorias},
             "modelo": {"sentimento": r.pred_sentimento, "categorias": r.pred_categorias}}
            for r in sorteio.itertuples()
        ],
    }
    (RESULTADOS / "analise_erros.json").write_text(json.dumps(saida, ensure_ascii=False, indent=1))
    print(f"{saida['n_erros']}/{saida['n_teste']} com erro (ref.: {nome_ref}); "
          f"só categorias {saida['so_categorias']}, só sentimento {saida['so_sentimento']}")
    for e in saida["exemplos"]:
        print(f"\n[{e['review_id'][:8]}] nota {e['nota']}, atraso_real={e['atraso_real']}\n  {e['texto']}\n"
              f"  ref: {e['referencia']['sentimento']} {e['referencia']['categorias']}\n"
              f"  mod: {e['modelo']['sentimento']} {e['modelo']['categorias']}")


if __name__ == "__main__":
    main()
