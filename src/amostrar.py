"""Etapa 3: amostragem e divisão treino/validação/teste.

Roda ANTES de qualquer leitura de texto: usa só a nota. A amostra é
estratificada por nota, com peso maior nas notas 1 a 3 (config.PESO_NOTA).
O teste é separado imediatamente e nunca é usado para decisões.

Uso: python src/amostrar.py
"""
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import (DATA_INTERIM, DATA_PROCESSED, FRACAO_VALIDACAO, N_AMOSTRA,
                    N_TESTE, PESO_NOTA, RESULTADOS, SEED, garantir_pastas)


def cotas_por_nota(contagem: pd.Series, n_total: int, pesos: dict) -> dict:
    """Divide n_total entre as notas proporcionalmente a contagem * peso."""
    massa = {nota: contagem[nota] * pesos[nota] for nota in contagem.index}
    soma = sum(massa.values())
    cotas = {nota: int(np.floor(n_total * m / soma)) for nota, m in massa.items()}
    # distribui o resto (arredondamento) para as notas de maior massa
    resto = n_total - sum(cotas.values())
    for nota in sorted(massa, key=massa.get, reverse=True)[:resto]:
        cotas[nota] += 1
    return {int(k): min(int(v), int(contagem[k])) for k, v in cotas.items()}


def amostrar(df: pd.DataFrame, seed: int = SEED):
    cotas = cotas_por_nota(df["nota"].value_counts().sort_index(), N_AMOSTRA, PESO_NOTA)
    partes = [df[df["nota"] == nota].sample(n=n, random_state=seed)
              for nota, n in cotas.items()]
    amostra = pd.concat(partes).sort_values("review_id").reset_index(drop=True)

    resto, teste = train_test_split(amostra, test_size=N_TESTE,
                                    stratify=amostra["nota"], random_state=seed)
    treino, validacao = train_test_split(resto, test_size=FRACAO_VALIDACAO,
                                         stratify=resto["nota"], random_state=seed)
    amostra["split"] = "treino"
    amostra.loc[amostra["review_id"].isin(validacao["review_id"]), "split"] = "validacao"
    amostra.loc[amostra["review_id"].isin(teste["review_id"]), "split"] = "teste"
    return amostra, cotas


def main():
    garantir_pastas()
    df = pd.read_parquet(DATA_INTERIM / "avaliacoes_com_texto.parquet")
    amostra, cotas = amostrar(df)
    amostra.to_parquet(DATA_PROCESSED / "amostra.parquet", index=False)
    resumo = {
        "seed": SEED,
        "cotas_por_nota": cotas,
        "tamanho_por_split": amostra["split"].value_counts().to_dict(),
        "nota_por_split": {s: g["nota"].value_counts().sort_index().to_dict()
                           for s, g in amostra.groupby("split")},
    }
    with open(RESULTADOS / "amostragem.json", "w", encoding="utf-8") as f:
        json.dump(resumo, f, ensure_ascii=False, indent=2, default=int)
    print(json.dumps(resumo, ensure_ascii=False, indent=2, default=int))


if __name__ == "__main__":
    main()
