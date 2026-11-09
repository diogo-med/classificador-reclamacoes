"""Etapa 12: reaplica a revisão manual e regenera métricas, tabelas e README.

Regra da planilha: coluna *_revisado(a) vazia = aceito o sugerido (rótulo do Gemini).

O que faz:
1. lê data/revisao/revisao_manual.csv e valida os valores preenchidos;
2. grava data/processed/rotulos_revisados.parquet (rótulos do Gemini com as
   correções humanas aplicadas); a partir daí src/avaliar.py usa esses rótulos;
3. mede a concordância Gemini x humano no teste e nos 150 sorteados do treino
   (o bloco de discordância NÃO entra na estimativa: é enviesado);
4. conta quantos rótulos de treino mudaram e, se algum mudou, recomenda retreinar
   e imprime os comandos do Kaggle;
5. roda src/avaliar.py.

Uso: python src/recalcular.py [--planilha CAMINHO] [--sem-avaliar]
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import CATEGORIAS, DATA_PROCESSED, DATA_REVISAO, RESULTADOS, SENTIMENTOS  # noqa: E402
from metricas import concordancia  # noqa: E402

SEP = ";"


def separar(celula: str) -> list[str]:
    return sorted({c.strip() for c in str(celula).replace(",", SEP).split(SEP) if c.strip()})


def aplicar_revisao(planilha: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Retorna (rótulos finais por review_id, lista de erros de preenchimento)."""
    erros, linhas = [], []
    for n, r in planilha.iterrows():
        linha_csv = n + 2  # cabeçalho + base 1
        sent = r["sentimento_revisado"].strip().lower() or r["sentimento_sugerido"].strip()
        cats = separar(r["categorias_revisadas"]) or separar(r["categorias_sugeridas"])
        if sent not in SENTIMENTOS:
            erros.append(f"linha {linha_csv} ({r['review_id']}): sentimento '{sent}' inválido")
        invalidas = [c for c in cats if c not in CATEGORIAS]
        if invalidas or not cats:
            erros.append(f"linha {linha_csv} ({r['review_id']}): categorias inválidas {invalidas or '(vazio)'}")
        linhas.append({"review_id": r["review_id"], "bloco": r["bloco"], "sentimento": sent, "categorias": cats,
                       "mudou": bool(r["sentimento_revisado"].strip() or r["categorias_revisadas"].strip())})
    return pd.DataFrame(linhas), erros


def concordancia_humano(gemini: pd.DataFrame, humano: pd.DataFrame) -> dict:
    """Gemini x humano por bloco; o bloco de discordância é reportado à parte e não estima erro."""
    saida = {}
    for bloco in ["teste", "treino_aleatorio", "treino_discordancia"]:
        h = humano[humano["bloco"] == bloco]
        if len(h):
            saida[bloco] = concordancia(gemini, h[["review_id", "sentimento", "categorias"]])
    if "treino_discordancia" in saida:
        saida["treino_discordancia"]["aviso"] = ("bloco enviesado (escolhido por discordância Gemini x 9B); "
                                                 "não usar para estimar a taxa de erro do Gemini")
    return saida


def comandos_retreino() -> list[str]:
    return [
        "python src/kaggle_dataset.py -m 'rótulos revisados'",
        "python kaggle/gerar_notebooks.py",
        "kaggle kernels push -p kaggle/treino_qwen2b_lora",
        "kaggle kernels status diogomede/cr-treino-qwen2b-lora   # repetir até COMPLETE",
        "kaggle kernels output diogomede/cr-treino-qwen2b-lora -p outputs/kaggle/treino_qwen2b_lora",
        "python src/importar_kaggle.py treino_qwen2b_lora",
        "python src/avaliar.py",
    ]


def recalcular(planilha_csv: Path, destino_dir: Path = DATA_PROCESSED, resultados_dir: Path = RESULTADOS) -> dict:
    planilha = pd.read_csv(planilha_csv, dtype=str, keep_default_na=False)
    humano, erros = aplicar_revisao(planilha)
    if erros:
        raise ValueError("Corrija a planilha:\n" + "\n".join(erros))
    gemini = pd.read_parquet(destino_dir / "rotulos_gemini.parquet")
    gemini["categorias"] = gemini["categorias"].apply(lambda c: sorted(list(c)))

    final = gemini.set_index("review_id").copy()
    for r in humano.itertuples():
        final.at[r.review_id, "sentimento"] = r.sentimento
        final.at[r.review_id, "categorias"] = r.categorias
    final = final.reset_index()
    final["revisado_por_humano"] = final["review_id"].isin(humano["review_id"])
    final.to_parquet(destino_dir / "rotulos_revisados.parquet", index=False)

    antes = gemini.set_index("review_id")
    treino_h = humano[humano["bloco"].str.startswith("treino")]
    mudaram_treino = sum(
        1 for r in treino_h.itertuples()
        if r.sentimento != antes.at[r.review_id, "sentimento"] or r.categorias != antes.at[r.review_id, "categorias"])
    relatorio = {
        "linhas_revisadas": int(len(humano)),
        "linhas_com_correcao": int(humano["mudou"].sum()),
        "rotulos_de_treino_alterados": int(mudaram_treino),
        "concordancia_gemini_humano": concordancia_humano(gemini, humano),
        "observacao": ("A revisão foi feita vendo o rótulo sugerido pelo Gemini: há viés de ancoragem, "
                       "e a concordância tende a ser superestimada."),
    }
    if mudaram_treino:
        relatorio["recomendacao"] = (f"{mudaram_treino} rótulos de treino mudaram: recomendo retreinar o 2B "
                                     "com os rótulos revisados.")
        relatorio["comandos_retreino"] = comandos_retreino()
    resultados_dir.mkdir(exist_ok=True)
    (resultados_dir / "revisao_humana.json").write_text(json.dumps(relatorio, ensure_ascii=False, indent=2))
    return relatorio


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--planilha", default=str(DATA_REVISAO / "revisao_manual.csv"))
    p.add_argument("--sem-avaliar", action="store_true")
    a = p.parse_args()
    rel = recalcular(Path(a.planilha))
    for bloco, c in rel["concordancia_gemini_humano"].items():
        print(f"Gemini x humano [{bloco}] n={c['n']}: sentimento {c['sentimento']['concordancia_pct']:.1f}%, "
              f"conjunto de categorias idêntico {c['conjunto_identico_pct']:.1f}%, "
              f"kappa médio {c['kappa_medio_categorias']:.2f}")
    print(f"{rel['linhas_com_correcao']} de {rel['linhas_revisadas']} linhas corrigidas; "
          f"{rel['rotulos_de_treino_alterados']} rótulos de treino alterados.")
    if "recomendacao" in rel:
        print("\n" + rel["recomendacao"] + " Comandos:\n  " + "\n  ".join(rel["comandos_retreino"]))
    if not a.sem_avaliar:
        import avaliar
        avaliar.main()


if __name__ == "__main__":
    main()
