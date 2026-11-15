"""Etapa 12: reaplica a revisão manual e regenera métricas, tabelas e README.

Regra da planilha: coluna *_revisado(a) vazia = aceito o sugerido (rótulo do Gemini).
Coluna `conferido`: "x" em cada linha que o revisor de fato leu. Só linhas com x entram:
- bloco `teste`: TODAS as linhas precisam de x; senão o script recusa rodar e diz quantas faltam;
- blocos de treino: linhas sem x são ignoradas (o rótulo do Gemini fica como está e a linha não
  entra na concordância Gemini x humano);
- linha com correção ou observação preenchida mas sem x é tratada como erro de preenchimento.

O que faz:
1. lê data/revisao/revisao_manual.csv e valida os valores preenchidos;
2. grava data/processed/rotulos_revisados.parquet (rótulos do Gemini com as
   correções humanas aplicadas); a partir daí src/avaliar.py usa esses rótulos;
3. mede a concordância Gemini x humano no teste e nos 150 sorteados do treino
   (o bloco de discordância NÃO entra na estimativa: é enviesado);
4. decide se vale retreinar o 2B (regra abaixo) e imprime os comandos do Kaggle;
5. roda src/avaliar.py.

Regra de retreino (fixada antes de ver os dados da revisão). Retreinar custa fila e GPU no
Kaggle e ~1 h de inferência local, então uma correção isolada não justifica. Estimamos a taxa
de erro do Gemini no treino só pelo bloco `treino_aleatorio` (amostra aleatória do treino),
com as linhas conferidas: "alterada" = sentimento ou conjunto de categorias final diferente
do rótulo do Gemini. Intervalo de confiança de 95% de Wilson. Decisão:
- menos de MIN_CONFERIDAS_RETREINO (50) linhas conferidas no bloco: "amostra insuficiente";
- taxa >= LIMIAR_ALTERACAO_RETREINO (10%): retreino recomendado;
- taxa abaixo do limiar: retreino opcional.
O bloco `treino_discordancia` não entra na taxa (foi escolhido por discordância Gemini x 9B,
logo superestima o erro), mas suas correções entram nos rótulos revisados e na contagem
`rotulos_de_treino_alterados`. Os comandos de retreino são sempre listados (como opcionais
quando a regra não recomenda).

Uso: python src/recalcular.py [--planilha CAMINHO] [--sem-avaliar]
"""
import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import CATEGORIAS, DATA_PROCESSED, DATA_REVISAO, RESULTADOS, SENTIMENTOS  # noqa: E402
from metricas import concordancia  # noqa: E402

SEP = ";"
MARCAS_CONFERIDO = {"x"}
BLOCOS = ["teste", "treino_aleatorio", "treino_discordancia"]
# regra de retreino (ver docstring)
BLOCO_ESTIMATIVA = "treino_aleatorio"
MIN_CONFERIDAS_RETREINO = 50
LIMIAR_ALTERACAO_RETREINO = 0.10
Z_95 = 1.959964


class RevisaoIncompleta(ValueError):
    """O bloco de teste tem linhas sem x em `conferido`."""


def checar_conferido(planilha: pd.DataFrame) -> tuple[pd.Series, list[str]]:
    """Devolve (máscara das linhas conferidas, erros de preenchimento).
    Levanta RevisaoIncompleta se faltar x em alguma linha do teste."""
    if "conferido" not in planilha.columns:
        raise ValueError("A planilha não tem a coluna 'conferido'. Gere de novo com python src/revisao.py "
                         "(ou acrescente a coluna e marque x nas linhas revisadas).")
    marca = planilha["conferido"].str.strip().str.lower()
    erros = []
    for n, (m, r) in enumerate(zip(marca, planilha.itertuples())):
        linha_csv = n + 2
        if m not in MARCAS_CONFERIDO | {""}:
            erros.append(f"linha {linha_csv} ({r.review_id}): 'conferido' deve ser x ou vazio, não '{r.conferido}'")
        elif m == "" and (r.sentimento_revisado.strip() or r.categorias_revisadas.strip() or r.observacao.strip()):
            erros.append(f"linha {linha_csv} ({r.review_id}): tem correção ou observação mas não tem x em 'conferido'")
    conferida = marca.isin(MARCAS_CONFERIDO)
    teste = planilha["bloco"] == "teste"
    faltam = int((teste & ~conferida).sum())
    if faltam:
        msg = (f"Faltam {faltam} de {int(teste.sum())} linhas do bloco 'teste' sem x na coluna 'conferido'. "
               "Todo o teste precisa ser conferido antes de recalcular.")
        raise RevisaoIncompleta("\n".join([msg] + (["Além disso, corrija:"] + erros if erros else [])))
    return conferida, erros


def contagem_conferidas(planilha: pd.DataFrame, conferida: pd.Series) -> dict:
    return {b: {"conferidas": int((conferida & (planilha["bloco"] == b)).sum()),
                "total": int((planilha["bloco"] == b).sum())}
            for b in BLOCOS if (planilha["bloco"] == b).any()}


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
        # mudou = rótulo final diferente do sugerido (célula preenchida com o mesmo valor não conta)
        mudou = (sent != r["sentimento_sugerido"].strip().lower()
                 or cats != separar(r["categorias_sugeridas"]))
        linhas.append({"review_id": r["review_id"], "bloco": r["bloco"], "sentimento": sent, "categorias": cats,
                       "mudou": bool(mudou)})
    return pd.DataFrame(linhas, columns=["review_id", "bloco", "sentimento", "categorias", "mudou"]), erros


def wilson(k: int, n: int, z: float = Z_95) -> tuple[float, float] | None:
    """Intervalo de confiança de Wilson para a proporção k/n (None se n = 0)."""
    if n <= 0:
        return None
    p = k / n
    centro = (p + z * z / (2 * n)) / (1 + z * z / n)
    meia = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centro - meia), min(1.0, centro + meia)


def decidir_retreino(alteradas: int, n: int) -> dict:
    """Aplica a regra de retreino à taxa de alteração do bloco treino_aleatorio."""
    taxa = alteradas / n if n else None
    ic = wilson(alteradas, n)
    if n < MIN_CONFERIDAS_RETREINO:
        decisao = "amostra_insuficiente"
    elif taxa >= LIMIAR_ALTERACAO_RETREINO:
        decisao = "retreinar"
    else:
        decisao = "opcional"
    taxa_txt = (f"{alteradas} de {n} rótulos alterados ({taxa:.1%}, IC 95% de Wilson {ic[0]:.1%} a {ic[1]:.1%})"
                if n else "nenhuma linha conferida")
    texto = {
        "retreinar": f"Retreino recomendado: taxa de alteração no {BLOCO_ESTIMATIVA} >= "
                     f"{LIMIAR_ALTERACAO_RETREINO:.0%}: {taxa_txt}.",
        "opcional": f"Retreino opcional: taxa de alteração no {BLOCO_ESTIMATIVA} abaixo de "
                    f"{LIMIAR_ALTERACAO_RETREINO:.0%}: {taxa_txt}.",
        "amostra_insuficiente": f"Amostra insuficiente para decidir: {n} linhas conferidas no {BLOCO_ESTIMATIVA} "
                                f"(mínimo {MIN_CONFERIDAS_RETREINO}); {taxa_txt}. Retreino opcional.",
    }[decisao]
    return {"bloco": BLOCO_ESTIMATIVA, "n_conferidas": int(n), "alteradas": int(alteradas),
            "taxa_alteracao": taxa, "ic95_wilson": list(ic) if ic else None,
            "limiar_taxa": LIMIAR_ALTERACAO_RETREINO, "min_conferidas": MIN_CONFERIDAS_RETREINO,
            "decisao": decisao, "texto": texto}


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
    conferida, erros_conf = checar_conferido(planilha)
    if erros_conf:
        raise ValueError("Corrija a planilha:\n" + "\n".join(erros_conf))
    conferidas = contagem_conferidas(planilha, conferida)
    planilha = planilha[conferida]  # linhas sem x não entram (índice original = linha do CSV)
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
    # alterada = rótulo final diferente do rótulo do Gemini
    alterada = pd.Series(
        [r.sentimento != antes.at[r.review_id, "sentimento"]
         or list(r.categorias) != list(antes.at[r.review_id, "categorias"]) for r in treino_h.itertuples()],
        index=treino_h.index, dtype=bool)
    mudaram_treino = int(alterada.sum())
    estimativa = treino_h["bloco"] == BLOCO_ESTIMATIVA
    retreino = decidir_retreino(int(alterada[estimativa].sum()), int(estimativa.sum()))
    relatorio = {
        "linhas_revisadas": int(len(humano)),
        "conferidas_por_bloco": conferidas,
        "linhas_com_correcao": int(humano["mudou"].sum()),
        "rotulos_de_treino_alterados": int(mudaram_treino),
        "concordancia_gemini_humano": concordancia_humano(gemini, humano),
        "observacao": ("A revisão foi feita vendo o rótulo sugerido pelo Gemini: há viés de ancoragem, "
                       "e a concordância tende a ser superestimada."),
    }
    relatorio["estimativa_erro_gemini_treino"] = retreino
    relatorio["retreino_recomendado"] = retreino["decisao"] == "retreinar"
    relatorio["recomendacao"] = retreino["texto"]
    relatorio["comandos_retreino"] = comandos_retreino()
    resultados_dir.mkdir(exist_ok=True)
    (resultados_dir / "revisao_humana.json").write_text(json.dumps(relatorio, ensure_ascii=False, indent=2))
    return relatorio


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--planilha", default=str(DATA_REVISAO / "revisao_manual.csv"))
    p.add_argument("--sem-avaliar", action="store_true")
    a = p.parse_args()
    try:
        rel = recalcular(Path(a.planilha))
    except ValueError as e:  # inclui RevisaoIncompleta
        sys.exit(str(e))
    for bloco, c in rel["conferidas_por_bloco"].items():
        print(f"conferidas [{bloco}]: {c['conferidas']} de {c['total']}")
    for bloco, c in rel["concordancia_gemini_humano"].items():
        print(f"Gemini x humano [{bloco}] n={c['n']}: sentimento {c['sentimento']['concordancia_pct']:.1f}%, "
              f"conjunto de categorias idêntico {c['conjunto_identico_pct']:.1f}%, "
              f"kappa médio {c['kappa_medio_categorias']:.2f}")
    print(f"{rel['linhas_com_correcao']} de {rel['linhas_revisadas']} linhas corrigidas; "
          f"{rel['rotulos_de_treino_alterados']} rótulos de treino alterados.")
    print("\n" + rel["recomendacao"])
    rotulo = "Comandos para retreinar" if rel["retreino_recomendado"] else "Comandos (opcionais) para retreinar"
    print(f"{rotulo}:\n  " + "\n  ".join(rel["comandos_retreino"]))
    if not a.sem_avaliar:
        import avaliar
        avaliar.main()
        imprimir_intervalos(RESULTADOS / "bootstrap_f1_macro.json")


def imprimir_intervalos(caminho: Path):
    """Resume os ICs de 95% (bootstrap pareado do teste) recalculados contra os rótulos revisados."""
    if not caminho.exists():
        return
    b = json.loads(caminho.read_text(encoding="utf-8"))
    print(f"\nF1 macro no teste com IC 95% (referência: {b.get('referencia')}, "
          f"{b['n_reamostras']} reamostragens, seed {b['seed']}):")
    for nome, r in b["modelos"].items():
        print(f"  {nome}: {r['f1_macro']:.3f} [{r['ic95'][0]:.3f}, {r['ic95'][1]:.3f}]")
    for par, d in b["diferencas"].items():
        conclusao = ("intervalo não inclui 0" if d["ic95"][0] > 0 or d["ic95"][1] < 0
                     else "intervalo inclui 0: diferença não conclusiva")
        print(f"  {par}: {d['diferenca_f1_macro']:+.3f} [{d['ic95'][0]:+.3f}, {d['ic95'][1]:+.3f}] ({conclusao})")


if __name__ == "__main__":
    main()
