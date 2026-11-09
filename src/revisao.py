"""Etapa 7: gera a planilha de revisão manual.

Ordem das linhas: os 500 do teste, 150 do treino sorteados, 150 do treino em
que Gemini e Qwen3.5-9B discordam (conjunto de categorias ou sentimento).
O atraso real (datas) NÃO aparece: ele é a checagem independente.

Uso: python src/revisao.py
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATA_PROCESSED, DATA_REVISAO, SEED  # noqa: E402

N_ALEATORIO = 150
N_DISCORDANCIA = 150
SEP = ";"  # separador das categorias dentro da célula
COLUNAS = ["review_id", "bloco", "nota", "texto", "sentimento_sugerido", "categorias_sugeridas",
           "sentimento_revisado", "categorias_revisadas", "observacao"]


def discordam(a: pd.DataFrame, b: pd.DataFrame) -> set:
    m = a.merge(b, on="review_id", suffixes=("_a", "_b"))
    dif = (m["sentimento_a"] != m["sentimento_b"]) | (
        m["categorias_a"].apply(lambda c: tuple(sorted(c))) != m["categorias_b"].apply(lambda c: tuple(sorted(c))))
    return set(m.loc[dif, "review_id"])


def montar_planilha(amostra, gemini, r9b, seed=SEED) -> pd.DataFrame:
    g = amostra.merge(gemini[["review_id", "sentimento", "categorias"]], on="review_id", how="left")
    teste = g[g["split"] == "teste"].sort_values("review_id").assign(bloco="teste")
    treino = g[(g["split"] == "treino") & g["sentimento"].notna()]
    aleatorio = treino.sample(n=min(N_ALEATORIO, len(treino)), random_state=seed).assign(bloco="treino_aleatorio")
    blocos = [teste, aleatorio]
    if r9b is not None:
        candidatos = treino[treino["review_id"].isin(discordam(gemini, r9b))
                            & ~treino["review_id"].isin(aleatorio["review_id"])]
        blocos.append(candidatos.sample(n=min(N_DISCORDANCIA, len(candidatos)), random_state=seed)
                      .assign(bloco="treino_discordancia"))
    df = pd.concat(blocos)
    return pd.DataFrame({
        "review_id": df["review_id"], "bloco": df["bloco"], "nota": df["nota"], "texto": df["texto"],
        "sentimento_sugerido": df["sentimento"].fillna(""),
        "categorias_sugeridas": df["categorias"].apply(lambda c: SEP.join(c) if isinstance(c, list) else ""),
        "sentimento_revisado": "", "categorias_revisadas": "", "observacao": "",
    })[COLUNAS]


def main():
    amostra = pd.read_parquet(DATA_PROCESSED / "amostra.parquet")
    gemini = pd.read_parquet(DATA_PROCESSED / "rotulos_gemini.parquet")
    gemini["categorias"] = gemini["categorias"].apply(list)
    caminho9 = DATA_PROCESSED / "rotulos_9b.parquet"
    r9b = None
    if caminho9.exists():
        r9b = pd.read_parquet(caminho9)
        r9b["categorias"] = r9b["categorias"].apply(list)
    destino = DATA_REVISAO / "revisao_manual.csv"
    if destino.exists():
        atual = pd.read_csv(destino, dtype=str, keep_default_na=False)
        if (atual[["sentimento_revisado", "categorias_revisadas", "observacao"]] != "").any().any():
            sys.exit(f"{destino} já tem revisões preenchidas; não vou sobrescrever.")
    planilha = montar_planilha(amostra, gemini, r9b)
    DATA_REVISAO.mkdir(parents=True, exist_ok=True)
    planilha.to_csv(destino, index=False, encoding="utf-8")
    print(planilha["bloco"].value_counts().to_dict(), "->", destino)


if __name__ == "__main__":
    main()
