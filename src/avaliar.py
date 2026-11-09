"""Etapas 8, 9 e 11: avaliação de todos os modelos no teste e regeneração de resultados/.

Entradas (todas no git):
- data/processed/amostra.parquet ............ splits e atraso real
- data/processed/rotulos_gemini.parquet ..... rótulos do Gemini (provisórios)
- data/processed/rotulos_revisados.parquet .. rótulos após revisão humana (se existir;
                                               gerado por src/recalcular.py)
- data/processed/rotulos_9b.parquet ......... segundo rotulador
- resultados/predicoes/*.jsonl .............. respostas cruas de cada modelo no teste
- resultados/eficiencia.json ................ medições de latência (src/eficiencia.py)

Saídas: resultados/metricas.json, tabelas em Markdown, checagem de atraso,
concordância Gemini x 9B, matriz de confusão e as tabelas do README.

Uso: python src/avaliar.py
"""
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.multiclass import OneVsRestClassifier  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import CATEGORIAS, DATA_PROCESSED, RAIZ, RESULTADOS, SEED  # noqa: E402
from esquema import avaliar_resposta  # noqa: E402
from metricas import (binarizar, bootstrap_f1_macro, checagem_atraso,  # noqa: E402
                      concordancia, metricas_categorias, metricas_formato,
                      metricas_sentimento)

PREDICOES = RESULTADOS / "predicoes"

# nome exibido -> (arquivo de previsões no teste, observação)
# O treino do Qwen3.5-0.8B só valida o pipeline (rótulos parciais) e NÃO entra aqui.
MODELOS = {
    "Qwen3.5-2B sem ajuste (Ollama q8_0)": ("qwen2b_teste.jsonl", ""),
    "Qwen3.5-9B sem ajuste (Ollama Q4_K_M)": ("qwen9b_teste.jsonl", ""),
    "Gemini 3.6 Flash (prompt de baseline)": ("gemini_teste.jsonl", "favorecida"),
    "Qwen3.5-2B + LoRA (este projeto)": ("qwen2b_lora_teste.jsonl", ""),
    "Qwen3.5-2B + LoRA, GGUF q8_0 no Ollama": ("qwen2b_lora_ollama_teste.jsonl", ""),
}
# Linha do professor: os próprios rótulos do Gemini comparados aos rótulos humanos. Antes da
# revisão, compará-los com eles mesmos daria F1 = 1,0 por definição; por isso a linha mostra
# "pendente de revisão" até src/recalcular.py rodar com os rótulos humanos.
LINHA_GEMINI = "Gemini (prompt de rotulagem, lotes de 100) vs. humano"
PRINCIPAL = "Qwen3.5-2B + LoRA (este projeto)"
NOVE_B = "Qwen3.5-9B sem ajuste (Ollama Q4_K_M)"
TFIDF = "TF-IDF + regressão logística"
N_BOOTSTRAP = 1000
# Aviso obrigatório enquanto a referência for o Gemini (ver README, "Leitura dos resultados").
NOTA_INFLADA = ("**Enquanto a referência for o Gemini, a vantagem do 2B ajustado está inflada**: ele foi "
                "treinado para imitar o estilo de rotulagem do Gemini, e esta tabela mede concordância com o "
                "Gemini. O 9B e o TF-IDF não tiveram esse benefício na mesma medida (o TF-IDF também foi "
                "treinado com rótulos do Gemini, mas com muito menos capacidade). Esta tabela **não** mostra "
                "que o 2B ajustado supera o 9B; isso só pode ser afirmado depois de `src/recalcular.py` "
                "rodar com os rótulos revisados por humano.")
NOTA_FAVORECIDA = ("favorecida por construção enquanto os rótulos do teste não forem revisados "
                   "(os rótulos provisórios vêm do próprio Gemini)")


# ------------------------------------------------------------------ dados
def ler_rotulos(nome: str) -> pd.DataFrame | None:
    caminho = DATA_PROCESSED / f"{nome}.parquet"
    if not caminho.exists():
        return None
    df = pd.read_parquet(caminho)
    df["categorias"] = df["categorias"].apply(lambda c: sorted(list(c)))
    return df


def rotulos_referencia() -> tuple[pd.DataFrame, bool]:
    """Rótulos revisados por humano, se existirem; senão, os do Gemini (provisórios)."""
    revisados = ler_rotulos("rotulos_revisados")
    if revisados is not None:
        return revisados, True
    return ler_rotulos("rotulos_gemini"), False


def ler_predicoes(arquivo: str) -> pd.DataFrame | None:
    caminho = PREDICOES / arquivo
    if not caminho.exists():
        return None
    linhas = [json.loads(l) for l in caminho.read_text(encoding="utf-8").splitlines()]
    df = pd.DataFrame(linhas)
    av = pd.DataFrame([avaliar_resposta(r) for r in df["resposta"]])
    return pd.concat([df[["review_id"]], av], axis=1)


# ------------------------------------------------------------------ TF-IDF
def treinar_tfidf(treino: pd.DataFrame, validacao: pd.DataFrame):
    """TF-IDF + regressão logística um-contra-todos (categorias) e multinomial (sentimento).
    C é escolhido na validação (F1 macro); o teste não participa."""
    vet = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, strip_accents="unicode")
    xt, xv = vet.fit_transform(treino["texto"]), vet.transform(validacao["texto"])
    yt = binarizar(treino["categorias"])
    # categoria sem nenhum exemplo positivo no treino não pode ser aprendida
    ativas = [i for i in range(len(CATEGORIAS)) if 0 < yt[:, i].sum() < len(yt)]
    melhor = None
    for c in [0.3, 1, 3, 10]:
        clf = OneVsRestClassifier(LogisticRegression(C=c, max_iter=2000, class_weight="balanced"))
        clf.fit(xt, yt[:, ativas])
        prev = prever_categorias(clf, xv, ativas)
        f1 = metricas_categorias(validacao["categorias"], prev)["f1_macro"]
        if melhor is None or f1 > melhor[0]:
            melhor = (f1, c, clf)
    sent = LogisticRegression(C=melhor[1], max_iter=2000, class_weight="balanced")
    sent.fit(xt, treino["sentimento"])
    return {"vet": vet, "cat": melhor[2], "sent": sent, "ativas": ativas,
            "C": melhor[1], "f1_macro_validacao": melhor[0]}


def prever_categorias(clf, x, ativas) -> list[list[str]]:
    prob = clf.predict_proba(x)
    saida = []
    for linha in prob:
        cats = [CATEGORIAS[ativas[j]] for j, p in enumerate(linha) if p >= 0.5]
        if not cats:  # sempre pelo menos uma categoria, como os LLMs
            cats = [CATEGORIAS[ativas[int(np.argmax(linha))]]]
        saida.append(sorted(cats))
    return saida


def predicoes_tfidf(modelo, teste: pd.DataFrame) -> pd.DataFrame:
    x = modelo["vet"].transform(teste["texto"])
    return pd.DataFrame({"review_id": teste["review_id"].values,
                         "json_valido": True, "no_esquema": True,
                         "sentimento": modelo["sent"].predict(x),
                         "categorias": prever_categorias(modelo["cat"], x, modelo["ativas"])})


# ------------------------------------------------------------------ métricas
def avaliar_modelo(pred: pd.DataFrame, ouro: pd.DataFrame) -> dict:
    """ouro: review_id, sentimento, categorias, atraso_real (só o teste)."""
    m = ouro.merge(pred, on="review_id", how="left", suffixes=("", "_p"))
    faltando = int(m["json_valido"].isna().sum())
    m["json_valido"] = m["json_valido"].fillna(False).astype(bool)
    m["no_esquema"] = m["no_esquema"].fillna(False).astype(bool)
    m["categorias_p"] = m["categorias_p"].apply(lambda c: c if isinstance(c, list) else [])
    return {
        "_categorias_previstas": m["categorias_p"].tolist(),  # alinhado com `ouro`; só para o bootstrap
        "n": int(len(m)), "sem_previsao": faltando,
        **metricas_formato(m["json_valido"], m["no_esquema"]),
        "categorias": metricas_categorias(m["categorias"], m["categorias_p"]),
        "sentimento": metricas_sentimento(m["sentimento"], m["sentimento_p"].tolist()),
        "checagem_atraso": checagem_atraso(m["atraso_real"], m["categorias_p"]),
    }


def nota_lotes_teste() -> str:
    """Quantas avaliações do teste o Gemini rotulou em cada tamanho de lote."""
    g = ler_rotulos("rotulos_gemini")
    if g is None or "tamanho_lote" not in g.columns:
        return ""
    sp = pd.read_parquet(DATA_PROCESSED / "amostra.parquet").set_index("review_id")["split"]
    cont = g[g["review_id"].map(sp) == "teste"]["tamanho_lote"].value_counts().sort_index()
    partes = ", ".join(f"{v} em lotes de {int(k)}" for k, v in cont.items())
    return f"\nRótulos do Gemini no teste: {partes} (ver DECISOES.md D19).\n"


def fmt(x, casas=3):
    return "—" if x is None else f"{x:.{casas}f}"


def ic(boot: dict | None, nome: str) -> str:
    r = (boot or {}).get("modelos", {}).get(nome)
    return "—" if r is None else f"[{r['ic95'][0]:.3f}, {r['ic95'][1]:.3f}]"


def texto_diferencas(boot: dict | None, revisado: bool = False) -> str:
    if not boot or not boot.get("diferencas"):
        return ""
    contra = ("rótulos revisados por humano" if revisado else
              "rótulos **provisórios do Gemini**, ou seja, diferença de concordância com o Gemini")
    linhas = [f"\nDiferenças de F1 macro com IC de 95% (bootstrap pareado, {boot['n_reamostras']} reamostragens "
              f"do teste com reposição, seed {boot['seed']}, percentis 2,5 e 97,5), medidas contra {contra}:\n"]
    for par, d in boot["diferencas"].items():
        linhas.append(f"- {par}: {d['diferenca_f1_macro']:+.3f} [{d['ic95'][0]:+.3f}, {d['ic95'][1]:+.3f}]; "
                      f"primeiro maior em {100 * d['fracao_reamostras_a_maior']:.1f}% das reamostragens")
    linhas.append("\nO intervalo cobre só a variação por amostragem das 500 avaliações do teste; não cobre erros "
                  "dos rótulos de referência nem a variação entre treinos (um só seed).")
    return "\n".join(linhas) + "\n"


def tabela_qualidade(res: dict, revisado: bool, boot: dict | None = None) -> str:
    cab = ("| Modelo | F1 macro | F1 macro IC 95% | F1 micro | F1 `atraso` | Acerto exato | Sentimento (acc / F1 macro) "
           "| JSON válido (%) | JSON no esquema (%) | Atraso real: cobertura (%) |\n"
           "|---|---|---|---|---|---|---|---|---|---|\n")
    linhas = []
    for nome, r in res.items():
        if r.get("pendente_de_revisao"):
            linhas.append(f"| {nome} | pendente de revisão |" + " — |" * 9)
            continue
        c, s = r["categorias"], r["sentimento"]
        marca = " ⚠️" if r.get("observacao") == "favorecida" and not revisado else ""
        linhas.append(
            f"| {nome}{marca} | {fmt(c['f1_macro'])} | {ic(boot, nome)} | {fmt(c['f1_micro'])} | "
            f"{fmt(c['por_categoria']['atraso']['f1'])} | {fmt(c['acerto_exato'])} | "
            f"{fmt(s['acuracia'])} / {fmt(s['f1_macro'])} | {r['json_valido_pct']:.1f} | "
            f"{r['json_no_esquema_pct']:.1f} | "
            f"{r['checagem_atraso']['atraso']['cobertura_dos_atrasados_pct']:.1f} |")
    favorecida = any(r.get("observacao") == "favorecida" for r in res.values())
    nota = texto_diferencas(boot, revisado)
    if not revisado:
        nota += f"\n{NOTA_INFLADA}\n"
    nota += f"\n⚠️ Linha {NOTA_FAVORECIDA}.\n" if favorecida and not revisado else ""
    if not revisado:
        nota += (f"\nA linha **{LINHA_GEMINI}** fica pendente: antes da revisão, os rótulos de referência "
                 "são os do próprio Gemini, e compará-los com eles mesmos daria F1 = 1,0 por definição. "
                 "Ela é preenchida por `python src/recalcular.py` depois da revisão humana.\n")
    nota += nota_lotes_teste()
    ref = ("rótulos revisados por humano" if revisado else
           "rótulos **provisórios** do Gemini (revisão humana pendente)")
    n = next(r["n"] for r in res.values() if "n" in r)
    return f"Teste: {n} avaliações; referência: {ref}.\n\n" + cab + "\n".join(linhas) + "\n" + nota


def tabela_f1_por_categoria(res: dict) -> str:
    cab = "| Modelo | " + " | ".join(f"`{c}`" for c in CATEGORIAS) + " |\n|---|" + "---|" * len(CATEGORIAS) + "\n"
    res = {k: v for k, v in res.items() if not v.get("pendente_de_revisao")}
    linhas = [f"| {nome} | " + " | ".join(fmt(r["categorias"]["por_categoria"][c]["f1"], 2) for c in CATEGORIAS) + " |"
              for nome, r in res.items()]
    suporte = next(iter(res.values()))["categorias"]["por_categoria"]
    linhas.append("| *suporte no teste* | " + " | ".join(str(suporte[c]["suporte"]) for c in CATEGORIAS) + " |")
    return cab + "\n".join(linhas) + "\n"


def tabela_eficiencia() -> str:
    caminho = RESULTADOS / "eficiencia.json"
    if not caminho.exists():
        return "_Medição de eficiência ainda não executada (`python src/eficiencia.py`)._\n"
    ef = json.loads(caminho.read_text(encoding="utf-8"))
    cab = ("| Modelo | Onde | n | Latência média (s) | Latência p95 (s) | Tokens/s (geração) | "
           "JSON válido (%) | Custo por 1.000 avaliações |\n|---|---|---|---|---|---|---|---|\n")
    linhas = []
    for nome, r in ef["modelos"].items():
        tps = fmt(r.get("tokens_por_segundo"), 1)
        linhas.append(f"| {nome} | {r['onde']} | {r['n']} | {fmt(r['latencia_media_s'], 2)} | "
                      f"{fmt(r['latencia_p95_s'], 2)} | {tps} | {fmt(r.get('json_valido_pct'), 1)} | "
                      f"{r.get('custo_por_1000', '—')} |")
    return (f"Hardware local: {ef['hardware']['resumo']}.\n\n" + cab + "\n".join(linhas) + "\n\n"
            + "\n".join(f"- {p}" for p in ef.get("premissas", [])) + "\n")


def grafico_matrizes(res: dict, nome: str, caminho: Path):
    pc = res[nome]["categorias"]["por_categoria"]
    fig, eixos = plt.subplots(2, 4, figsize=(13, 6.5))
    for ax, c in zip(eixos.ravel(), CATEGORIAS):
        mt = pc[c]["matriz"]
        mat = np.array([[mt["vn"], mt["fp"]], [mt["fn"], mt["vp"]]])
        ax.imshow(mat, cmap="Blues")
        for (i, j), v in np.ndenumerate(mat):
            ax.text(j, i, str(v), ha="center", va="center",
                    color="white" if v > mat.max() / 2 else "black", fontsize=12)
        ax.set_title(f"{c} (F1 {pc[c]['f1']:.2f})", fontsize=10)
        ax.set_xticks([0, 1], ["prev. 0", "prev. 1"])
        ax.set_yticks([0, 1], ["real 0", "real 1"])
    fig.suptitle(f"Matrizes 2×2 por categoria no teste — {nome}")
    fig.tight_layout()
    fig.savefig(caminho, dpi=110)
    plt.close(fig)


def atualizar_readme(blocos: dict):
    """Substitui o conteúdo entre <!-- INICIO:x --> e <!-- FIM:x --> no README."""
    caminho = RAIZ / "README.md"
    texto = caminho.read_text(encoding="utf-8")
    for chave, conteudo in blocos.items():
        padrao = re.compile(rf"(<!-- INICIO:{chave} -->\n).*?(<!-- FIM:{chave} -->)", re.DOTALL)
        texto = padrao.sub(lambda m: m.group(1) + conteudo + m.group(2), texto)
    caminho.write_text(texto, encoding="utf-8")


# ------------------------------------------------------------------ principal
def main():
    RESULTADOS.mkdir(exist_ok=True)
    amostra = pd.read_parquet(DATA_PROCESSED / "amostra.parquet")
    ref, revisado = rotulos_referencia()
    gemini = ler_rotulos("rotulos_gemini")
    dados = amostra.merge(ref[["review_id", "sentimento", "categorias"]], on="review_id")
    teste = dados[dados["split"] == "teste"].sort_values("review_id").reset_index(drop=True)
    saida = {"referencia": "revisada" if revisado else "provisoria_gemini",
             "n_rotulados_por_split": dados["split"].value_counts().to_dict(),
             "modelos": {}}

    # TF-IDF (treinado aqui mesmo)
    tfidf = treinar_tfidf(dados[dados["split"] == "treino"], dados[dados["split"] == "validacao"])
    res = {TFIDF: avaliar_modelo(predicoes_tfidf(tfidf, teste), teste)}
    res[TFIDF]["hiperparametros"] = {"C": tfidf["C"], "f1_macro_validacao": tfidf["f1_macro_validacao"]}
    for nome, (arquivo, obs) in MODELOS.items():
        pred = ler_predicoes(arquivo)
        if pred is None:
            print(f"(sem previsões para {nome}: {arquivo})")
            continue
        res[nome] = avaliar_modelo(pred, teste)
        res[nome]["observacao"] = obs
    # Linha do professor (Gemini, prompt de rotulagem) — só depois da revisão humana
    if revisado:
        g = gemini.assign(json_valido=True, no_esquema=True)[["review_id", "json_valido", "no_esquema",
                                                               "sentimento", "categorias"]]
        res[LINHA_GEMINI] = avaliar_modelo(g, teste)
    else:
        res[LINHA_GEMINI] = {"pendente_de_revisao": True}

    # Intervalos de confiança por bootstrap pareado (mesmas reamostragens para todos os modelos)
    previstas = {nome: r.pop("_categorias_previstas") for nome, r in res.items() if "_categorias_previstas" in r}
    pares = [(PRINCIPAL, NOVE_B), (PRINCIPAL, TFIDF)]
    if revisado:
        pares.append((PRINCIPAL, LINHA_GEMINI))
    boot = bootstrap_f1_macro(teste["categorias"].tolist(), previstas, pares=pares,
                              n_reamostras=N_BOOTSTRAP, seed=SEED)
    boot["referencia"] = saida["referencia"]
    saida["bootstrap_f1_macro"] = boot
    (RESULTADOS / "bootstrap_f1_macro.json").write_text(json.dumps(boot, ensure_ascii=False, indent=2))
    saida["modelos"] = res

    # Etapa 8: checagem independente dos rotuladores (em todas as avaliações com rótulo)
    checagem = {}
    for nome, rot in [("Gemini (rótulos)", gemini), ("Qwen3.5-9B (rótulos)", ler_rotulos("rotulos_9b"))]:
        if rot is None:
            continue
        m = amostra.merge(rot, on="review_id")
        checagem[nome] = {s: checagem_atraso(g["atraso_real"], g["categorias"])
                          for s, g in [("todas", m), ("teste", m[m["split"] == "teste"])]}
    for nome, r in res.items():
        if "checagem_atraso" in r:
            checagem[nome] = {"teste": r["checagem_atraso"]}
    saida["checagem_atraso"] = checagem

    # Etapa 6: concordância Gemini x 9B (mede ambiguidade; não usada para mudar nada)
    r9 = ler_rotulos("rotulos_9b")
    if r9 is not None:
        sp = amostra.set_index("review_id")["split"]
        saida["concordancia_gemini_9b"] = {
            s: concordancia(gemini[gemini["review_id"].map(sp) == s], r9[r9["review_id"].map(sp) == s])
            for s in ["teste", "treino"]}
        (RESULTADOS / "concordancia_gemini_9b.json").write_text(
            json.dumps(saida["concordancia_gemini_9b"], ensure_ascii=False, indent=2))

    (RESULTADOS / "metricas.json").write_text(json.dumps(saida, ensure_ascii=False, indent=2, default=str))
    (RESULTADOS / "checagem_atraso.json").write_text(json.dumps(checagem, ensure_ascii=False, indent=2))
    tq = tabela_qualidade(res, revisado, boot)
    tc = tabela_f1_por_categoria(res)
    te = tabela_eficiencia()
    tca = tabela_concordancia(saida.get("concordancia_gemini_9b"))
    tat = tabela_checagem(checagem)
    for nome, conteudo in [("tabela_qualidade.md", tq), ("tabela_f1_categorias.md", tc),
                           ("tabela_eficiencia.md", te), ("tabela_concordancia.md", tca),
                           ("tabela_checagem_atraso.md", tat)]:
        (RESULTADOS / nome).write_text(conteudo, encoding="utf-8")
    alvo = PRINCIPAL if PRINCIPAL in res else [k for k, v in res.items() if "categorias" in v][-1]
    grafico_matrizes(res, alvo, RESULTADOS / "matriz_confusao.png")
    atualizar_readme({"qualidade": tq, "f1_categorias": tc, "eficiencia": te,
                      "concordancia": tca, "checagem": tat})
    print(tq)
    print(te)


def tabela_concordancia(conc) -> str:
    if not conc:
        return "_Concordância Gemini × 9B ainda não calculada._\n"
    cab = "| Recorte | n | Sentimento (% / κ) | " + " | ".join(f"`{c}` (% / κ)" for c in CATEGORIAS) + " | Conjunto idêntico (%) |\n"
    cab += "|---|---|---|" + "---|" * len(CATEGORIAS) + "---|\n"
    linhas = []
    for s, r in conc.items():
        cats = " | ".join(f"{r['categorias'][c]['concordancia_pct']:.0f} / {r['categorias'][c]['kappa']:.2f}" for c in CATEGORIAS)
        linhas.append(f"| {s} | {r['n']} | {r['sentimento']['concordancia_pct']:.0f} / {r['sentimento']['kappa']:.2f} | "
                      f"{cats} | {r['conjunto_identico_pct']:.1f} |")
    return cab + "\n".join(linhas) + "\n"


def _direcoes(x: dict) -> tuple[str, str]:
    """(atrasados que o rótulo marca, marcados que de fato atrasaram), com números absolutos."""
    return (f"{x['marcados_e_atrasados']}/{x['atrasados_reais']} ({x['cobertura_dos_atrasados_pct']:.1f}%)",
            f"{x['marcados_e_atrasados']}/{x['marcados']} ({x['precisao_pct']:.1f}%)")


def tabela_checagem(checagem) -> str:
    cab = ("| Fonte | Recorte | n (atraso conhecido) | Atrasados reais | `atraso`: atrasados marcados | "
           "`atraso`: marcados que atrasaram | `atraso` ou `nao_recebido`: atrasados marcados | "
           "`atraso` ou `nao_recebido`: marcados que atrasaram | κ (`atraso`) |\n"
           "|---|---|---|---|---|---|---|---|---|\n")
    linhas = []
    for nome, recortes in checagem.items():
        for s, r in recortes.items():
            a, b = r["atraso"], r["atraso_ou_nao_recebido"]
            (a1, a2), (b1, b2) = _direcoes(a), _direcoes(b)
            linhas.append(f"| {nome} | {s} | {r['n']} | {a['atrasados_reais']} ({100 * r['taxa_atraso_real']:.1f}%) | "
                          f"{a1} | {a2} | {b1} | {b2} | {fmt(a['kappa'], 2)} |")
    nota = ("\n\"Atrasados marcados\" = dos pedidos que atrasaram pelas datas, quantos o rótulo marca (cobertura). "
            "\"Marcados que atrasaram\" = das avaliações com o rótulo, quantas tiveram atraso pelas datas "
            "(precisão).\n")
    antes = RESULTADOS / "avaliacao_antes_da_entrega.json"
    if antes.exists():
        d = json.loads(antes.read_text(encoding="utf-8"))["teste"]
        nota += (f"\nNo teste, {d['atrasados_avaliados_antes_da_entrega']} dos {d['atrasados_reais']} pedidos "
                 "atrasados pelas datas foram avaliados **antes** de serem entregues "
                 "(`resultados/avaliacao_antes_da_entrega.json`, gerado por `src/datas_avaliacao.py`). Nesses "
                 "casos o cliente escreve \"não recebi\", e o rótulo esperado é `nao_recebido` (com `atraso` só "
                 "se ele disser que o prazo passou). Por isso a cobertura de `atraso` sozinho é baixa para todos "
                 "os modelos, e a coluna `atraso` ou `nao_recebido` é a comparação mais justa.\n")
    return cab + "\n".join(linhas) + "\n" + nota


if __name__ == "__main__":
    np.random.seed(SEED)
    main()
