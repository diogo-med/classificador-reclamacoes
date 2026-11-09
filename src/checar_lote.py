"""Checagem de consistência do tamanho de lote na rotulagem do Gemini (DECISOES.md D29).

Usa SÓ avaliações de TREINO que já têm rótulo feito em lote de 25 (checkpoint de 08/11).
- Requisição A: 100 delas num único lote de 100      -> compara lote 25 x lote 100
- Requisição B: 25 delas de novo num lote de 25       -> piso de ruído (25 x 25)
Total: 2 requisições da cota diária.

Regra fixada antes de ver o resultado: se o "conjunto de categorias idêntico" entre lote 25 e
lote 100 ficar mais de 10 pontos percentuais abaixo do piso de ruído (25 x 25), a rotulagem
restante usa lotes de 50 em vez de 100.

Uso: python src/checar_lote.py
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATA_PROCESSED, OUTPUTS, RESULTADOS, SEED  # noqa: E402
from metricas import concordancia  # noqa: E402
from rotular import consolidar, rotular  # noqa: E402

LIMIAR_PP = 10


def main():
    amostra = pd.read_parquet(DATA_PROCESSED / "amostra.parquet")
    base = consolidar(OUTPUTS / "rotulos_gemini.jsonl")
    base = base[base["tamanho_lote"] == 25]
    treino_ids = set(amostra.loc[amostra["split"] == "treino", "review_id"])
    base = base[base["review_id"].isin(treino_ids)]   # nunca teste nem validação
    ids100 = base.sample(n=100, random_state=SEED)["review_id"].tolist()
    ids25 = ids100[:25]
    df = amostra.set_index("review_id")

    saida_a = OUTPUTS / "checagem_lote" / "lote100.jsonl"
    saida_b = OUTPUTS / "checagem_lote" / "lote25_repetido.jsonl"
    rotular(df.loc[ids100].reset_index(), "gemini", saida_a, tamanho_lote=100)
    rotular(df.loc[ids25].reset_index(), "gemini", saida_b, tamanho_lote=25)
    l100, l25b = consolidar(saida_a), consolidar(saida_b)
    for d in (base, l100, l25b):
        d["categorias"] = d["categorias"].apply(lambda c: sorted(list(c)))

    c_25_100 = concordancia(base[base["review_id"].isin(ids100)], l100)
    c_ruido = concordancia(base[base["review_id"].isin(ids25)], l25b)
    queda = c_ruido["conjunto_identico_pct"] - c_25_100["conjunto_identico_pct"]
    resultado = {
        "descricao": "Gemini 3.6 Flash, mesmas avaliações de treino rotuladas em lotes de 25 e de 100",
        "splits_usados": sorted(amostra.set_index("review_id").loc[ids100, "split"].unique().tolist()),
        "lote25_vs_lote100": c_25_100,
        "piso_de_ruido_lote25_vs_lote25": c_ruido,
        "queda_conjunto_identico_pp": queda,
        "regra": f"usar lotes de 50 se a queda passar de {LIMIAR_PP} p.p.",
        "decisao": "lotes de 50" if queda > LIMIAR_PP else "lotes de 100",
    }
    (RESULTADOS / "checagem_lote.json").write_text(json.dumps(resultado, ensure_ascii=False, indent=2))
    print(json.dumps({k: v for k, v in resultado.items() if not isinstance(v, dict)}, ensure_ascii=False, indent=1))
    print("25x100: conjunto idêntico %.1f%%, sentimento %.1f%%; ruído 25x25: %.1f%%, %.1f%%" % (
        c_25_100["conjunto_identico_pct"], c_25_100["sentimento"]["concordancia_pct"],
        c_ruido["conjunto_identico_pct"], c_ruido["sentimento"]["concordancia_pct"]))


if __name__ == "__main__":
    main()
