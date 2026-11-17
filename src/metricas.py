"""Funções de métrica. Sem I/O: recebem listas/DataFrames e devolvem dicionários.

Convenções:
- categorias são listas de nomes; viram vetores binários na ordem de CATEGORIAS;
- uma resposta inválida (JSON quebrado) tem categorias [] e sentimento None,
  e conta como erro (nenhuma categoria prevista, sentimento errado).
"""
import re
import unicodedata

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, cohen_kappa_score, confusion_matrix,
                             f1_score)

from config import CATEGORIAS, SENTIMENTOS


def binarizar(listas, categorias=CATEGORIAS) -> np.ndarray:
    """Lista de listas de categorias -> matriz 0/1 (n_exemplos x n_categorias)."""
    idx = {c: i for i, c in enumerate(categorias)}
    m = np.zeros((len(listas), len(categorias)), dtype=int)
    for linha, cats in enumerate(listas):
        for c in cats or []:
            if c in idx:
                m[linha, idx[c]] = 1
    return m


def metricas_categorias(verdade, previsto, categorias=CATEGORIAS, sample_weight=None) -> dict:
    """F1 binário por categoria, F1 macro e micro, acerto exato e matriz 2x2 por categoria.

    sample_weight (opcional): peso por avaliação. Com peso, os F1 são os ponderados do sklearn
    (macro = média dos F1 binários ponderados), o acerto exato é a média ponderada e "matriz" e
    "suporte" viram somas de pesos (float). Sem peso, tudo é contagem (int), como antes."""
    y, p = binarizar(verdade, categorias), binarizar(previsto, categorias)
    w = None if sample_weight is None else np.asarray(sample_weight, dtype=float)
    num = int if w is None else float
    por_cat = {}
    for i, c in enumerate(categorias):
        tn, fp, fn, tp = confusion_matrix(y[:, i], p[:, i], labels=[0, 1], sample_weight=w).ravel()
        por_cat[c] = {
            "f1": float(f1_score(y[:, i], p[:, i], zero_division=0, sample_weight=w)),
            "suporte": num(y[:, i].sum() if w is None else (w * y[:, i]).sum()),
            "matriz": {"vp": num(tp), "fp": num(fp), "fn": num(fn), "vn": num(tn)},
        }
    return {
        "f1_macro": float(f1_score(y, p, average="macro", zero_division=0, sample_weight=w)),
        "f1_micro": float(f1_score(y, p, average="micro", zero_division=0, sample_weight=w)),
        "acerto_exato": float(np.average((y == p).all(axis=1), weights=w)),
        "por_categoria": por_cat,
    }


def normalizar_texto(texto) -> str:
    """Minúsculas, sem acentos, tudo que não é [a-z0-9] vira espaço, espaços colapsados.
    Usado para achar textos repetidos entre treino e teste ("Muito bom!" == "muito bom")."""
    t = unicodedata.normalize("NFKD", "" if pd.isna(texto) else str(texto).lower())
    t = "".join(ch for ch in t if not unicodedata.combining(ch))
    return " ".join(re.sub(r"[^a-z0-9]+", " ", t).split())


def metricas_sentimento(verdade, previsto) -> dict:
    """Acurácia e F1 macro (3 classes). Previsão None conta como erro."""
    p = [s if s in SENTIMENTOS else "__invalido__" for s in previsto]
    return {
        "acuracia": float(accuracy_score(verdade, p)),
        "f1_macro": float(f1_score(verdade, p, labels=SENTIMENTOS, average="macro", zero_division=0)),
    }


def metricas_formato(json_valido, no_esquema) -> dict:
    return {"json_valido_pct": 100 * float(np.mean(json_valido)),
            "json_no_esquema_pct": 100 * float(np.mean(no_esquema))}


def kappa(a, b) -> float:
    """Kappa de Cohen; se ambos forem constantes e iguais, devolve 1.0 (concordância total)."""
    a, b = list(a), list(b)
    if len(set(a) | set(b)) == 1:
        return 1.0
    return float(cohen_kappa_score(a, b))


def concordancia(rotulos_a: pd.DataFrame, rotulos_b: pd.DataFrame, categorias=CATEGORIAS) -> dict:
    """Concordância entre dois rotuladores nas avaliações em comum.

    Cada DataFrame tem review_id, sentimento e categorias. Retorna % de
    concordância e kappa por categoria (binário) e no sentimento."""
    m = rotulos_a.merge(rotulos_b, on="review_id", suffixes=("_a", "_b"))
    ya, yb = binarizar(m["categorias_a"], categorias), binarizar(m["categorias_b"], categorias)
    por_cat = {c: {"concordancia_pct": 100 * float((ya[:, i] == yb[:, i]).mean()),
                   "kappa": kappa(ya[:, i], yb[:, i]),
                   "freq_a": int(ya[:, i].sum()), "freq_b": int(yb[:, i].sum())}
               for i, c in enumerate(categorias)}
    sa = m["sentimento_a"].fillna("__nulo__")
    sb = m["sentimento_b"].fillna("__nulo__")
    return {
        "n": int(len(m)),
        "sentimento": {"concordancia_pct": 100 * float((sa == sb).mean()), "kappa": kappa(sa, sb)},
        "categorias": por_cat,
        "kappa_medio_categorias": float(np.mean([v["kappa"] for v in por_cat.values()])),
        "conjunto_identico_pct": 100 * float((ya == yb).all(axis=1).mean()),
    }


def checagem_atraso(atraso_real, previsto_listas) -> dict:
    """Compara a categoria prevista 'atraso' (e 'atraso OU nao_recebido') com o
    atraso real calculado pelas datas. Só entram avaliações com atraso_real conhecido."""
    real = pd.Series(atraso_real).astype("Float64")
    ok = real.notna().to_numpy()
    r = real[ok].astype(int).to_numpy()
    listas = [l for l, k in zip(previsto_listas, ok) if k]
    saida = {"n": int(ok.sum()), "taxa_atraso_real": float(r.mean()) if len(r) else None}
    for nome, cats in [("atraso", {"atraso"}), ("atraso_ou_nao_recebido", {"atraso", "nao_recebido"})]:
        p = np.array([int(bool(cats & set(l or []))) for l in listas])
        vp = int(((p == 1) & (r == 1)).sum())
        saida[nome] = {
            "concordancia_pct": 100 * float((p == r).mean()) if len(r) else None,
            "kappa": kappa(r, p) if len(r) else None,
            # dos realmente atrasados, quantos o rótulo marca (recall)
            "cobertura_dos_atrasados_pct": 100 * vp / max(int(r.sum()), 1),
            # dos marcados, quantos estavam realmente atrasados (precisão)
            "precisao_pct": 100 * vp / max(int(p.sum()), 1),
            # números absolutos, para ler as duas direções sem porcentagem
            "atrasados_reais": int(r.sum()),
            "marcados": int(p.sum()),
            "marcados_e_atrasados": vp,
        }
    return saida


def _f1_macro_matrizes(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    """F1 macro de várias reamostragens de uma vez. y, p: (B, n, k) 0/1 -> (B,).
    Mesma convenção do sklearn com zero_division=0: categoria sem VP, FP e FN vale 0."""
    vp = (y & p).sum(axis=1)
    fp = ((1 - y) & p).sum(axis=1)
    fn = (y & (1 - p)).sum(axis=1)
    den = 2 * vp + fp + fn
    f1 = np.divide(2 * vp, den, out=np.zeros(den.shape, dtype=float), where=den > 0)
    return f1.mean(axis=1)


def bootstrap_f1_macro(verdade, previsoes: dict, pares=(), n_reamostras: int = 1000, seed: int = 42,
                       categorias=CATEGORIAS, bloco: int = 100) -> dict:
    """Intervalo de confiança de 95% (percentil) do F1 macro por bootstrap do teste.

    verdade: listas de categorias (uma por avaliação). previsoes: {modelo: listas alinhadas}.
    As MESMAS reamostragens são usadas para todos os modelos (bootstrap pareado), então a
    diferença entre dois modelos em `pares` é calculada reamostra a reamostra.
    O IC mede só a variação por amostragem do teste; não cobre o erro dos rótulos de referência."""
    y = binarizar(verdade, categorias)
    ps = {m: binarizar(l, categorias) for m, l in previsoes.items()}
    n = len(y)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_reamostras, n))
    amostras = {m: [] for m in ps}
    for i in range(0, n_reamostras, bloco):
        ib = idx[i:i + bloco]
        yb = y[ib]
        for m, p in ps.items():
            amostras[m].append(_f1_macro_matrizes(yb, p[ib]))
    amostras = {m: np.concatenate(v) for m, v in amostras.items()}

    def resumo(pontual, dist):
        lo, hi = np.percentile(dist, [2.5, 97.5])
        return {"f1_macro": float(pontual), "ic95": [float(lo), float(hi)]}

    saida = {"n": int(n), "n_reamostras": int(n_reamostras), "seed": int(seed), "metodo": "percentil, pareado",
             "modelos": {m: resumo(_f1_macro_matrizes(y[None], p[None])[0], amostras[m]) for m, p in ps.items()},
             "diferencas": {}}
    for a, b in pares:
        if a in ps and b in ps:
            d = amostras[a] - amostras[b]
            r = resumo(saida["modelos"][a]["f1_macro"] - saida["modelos"][b]["f1_macro"], d)
            saida["diferencas"][f"{a} − {b}"] = {
                "diferenca_f1_macro": r["f1_macro"], "ic95": r["ic95"],
                "fracao_reamostras_a_maior": float((d > 0).mean())}
    return saida
