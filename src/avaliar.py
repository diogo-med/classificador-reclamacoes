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
from metricas import (binarizar, checagem_atraso, concordancia,  # noqa: E402
                      metricas_categorias, metricas_formato, metricas_sentimento)

PREDICOES = RESULTADOS / "predicoes"

# nome exibido -> (arquivo de previsões no teste, observação)
MODELOS = {
    "Qwen3.5-2B sem ajuste (Ollama q8_0)": ("qwen2b_teste.jsonl", ""),
    "Qwen3.5-9B sem ajuste (Ollama Q4_K_M)": ("qwen9b_teste.jsonl", ""),
    "Gemini 3.6 Flash (prompt de baseline)": ("gemini_teste.jsonl", "favorecida"),
    "Qwen3.5-0.8B + LoRA (validação do pipeline)": ("qwen08b_lora_teste.jsonl", ""),
    "Qwen3.5-2B + LoRA (este projeto)": ("qwen2b_lora_teste.jsonl", ""),
    "Qwen3.5-2B + LoRA, GGUF q8_0 no Ollama": ("qwen2b_lora_ollama_teste.jsonl", ""),
}
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
        "n": int(len(m)), "sem_previsao": faltando,
        **metricas_formato(m["json_valido"], m["no_esquema"]),
        "categorias": metricas_categorias(m["categorias"], m["categorias_p"]),
        "sentimento": metricas_sentimento(m["sentimento"], m["sentimento_p"].tolist()),
        "checagem_atraso": checagem_atraso(m["atraso_real"], m["categorias_p"]),
    }


def fmt(x, casas=3):
    return "—" if x is None else f"{x:.{casas}f}"


def tabela_qualidade(res: dict, revisado: bool) -> str:
    cab = ("| Modelo | F1 macro | F1 micro | F1 `atraso` | Acerto exato | Sentimento (acc / F1 macro) "
           "| JSON válido (%) | JSON no esquema (%) | Atraso real: cobertura (%) |\n"
           "|---|---|---|---|---|---|---|---|---|\n")
    linhas = []
    for nome, r in res.items():
        c, s = r["categorias"], r["sentimento"]
        marca = " ⚠️" if r.get("observacao") == "favorecida" and not revisado else ""
        linhas.append(
            f"| {nome}{marca} | {fmt(c['f1_macro'])} | {fmt(c['f1_micro'])} | "
            f"{fmt(c['por_categoria']['atraso']['f1'])} | {fmt(c['acerto_exato'])} | "
            f"{fmt(s['acuracia'])} / {fmt(s['f1_macro'])} | {r['json_valido_pct']:.1f} | "
            f"{r['json_no_esquema_pct']:.1f} | "
            f"{r['checagem_atraso']['atraso']['cobertura_dos_atrasados_pct']:.1f} |")
    nota = "" if revisado else f"\n⚠️ Linha {NOTA_FAVORECIDA}.\n"
    ref = ("rótulos revisados por humano" if revisado else
           "rótulos **provisórios** do Gemini (revisão humana pendente)")
    return f"Teste: {next(iter(res.values()))['n']} avaliações; referência: {ref}.\n\n" + cab + "\n".join(linhas) + "\n" + nota


def tabela_f1_por_categoria(res: dict) -> str:
    cab = "| Modelo | " + " | ".join(f"`{c}`" for c in CATEGORIAS) + " |\n|---|" + "---|" * len(CATEGORIAS) + "\n"
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
    res = {"TF-IDF + regressão logística": avaliar_modelo(predicoes_tfidf(tfidf, teste), teste)}
    res["TF-IDF + regressão logística"]["hiperparametros"] = {"C": tfidf["C"], "f1_macro_validacao": tfidf["f1_macro_validacao"]}
    for nome, (arquivo, obs) in MODELOS.items():
        pred = ler_predicoes(arquivo)
        if pred is None:
            print(f"(sem previsões para {nome}: {arquivo})")
            continue
        res[nome] = avaliar_modelo(pred, teste)
        res[nome]["observacao"] = obs
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
    tq = tabela_qualidade(res, revisado)
    tc = tabela_f1_por_categoria(res)
    te = tabela_eficiencia()
    tca = tabela_concordancia(saida.get("concordancia_gemini_9b"))
    tat = tabela_checagem(checagem)
    for nome, conteudo in [("tabela_qualidade.md", tq), ("tabela_f1_categorias.md", tc),
                           ("tabela_eficiencia.md", te), ("tabela_concordancia.md", tca),
                           ("tabela_checagem_atraso.md", tat)]:
        (RESULTADOS / nome).write_text(conteudo, encoding="utf-8")
    principal = "Qwen3.5-2B + LoRA (este projeto)"
    alvo = principal if principal in res else list(res)[-1]
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


def tabela_checagem(checagem) -> str:
    cab = ("| Fonte | Recorte | n (atraso conhecido) | Atrasados reais (%) | `atraso`: cobertura / precisão (%) | "
           "`atraso` ou `nao_recebido`: cobertura / precisão (%) | κ (`atraso`) |\n|---|---|---|---|---|---|---|\n")
    linhas = []
    for nome, recortes in checagem.items():
        for s, r in recortes.items():
            a, b = r["atraso"], r["atraso_ou_nao_recebido"]
            linhas.append(f"| {nome} | {s} | {r['n']} | {100 * r['taxa_atraso_real']:.1f} | "
                          f"{a['cobertura_dos_atrasados_pct']:.1f} / {a['precisao_pct']:.1f} | "
                          f"{b['cobertura_dos_atrasados_pct']:.1f} / {b['precisao_pct']:.1f} | {fmt(a['kappa'], 2)} |")
    return cab + "\n".join(linhas) + "\n"


if __name__ == "__main__":
    np.random.seed(SEED)
    main()
