import pandas as pd
import pytest

from metricas import (checagem_atraso, concordancia, metricas_categorias,
                      metricas_sentimento)

CATS = ["atraso", "elogio", "outro"]


def test_metricas_categorias_exemplo_feito_a_mao():
    verdade = [["atraso"], ["elogio"], ["atraso", "outro"], ["outro"]]
    previsto = [["atraso"], ["atraso"], ["atraso"], []]
    m = metricas_categorias(verdade, previsto, CATS)
    # atraso: VP=2, FP=1, FN=0 -> P=2/3, R=1, F1=0.8
    assert m["por_categoria"]["atraso"]["f1"] == pytest.approx(0.8)
    assert m["por_categoria"]["atraso"]["matriz"] == {"vp": 2, "fp": 1, "fn": 0, "vn": 1}
    # elogio: VP=0, FN=1 -> F1=0 ; outro: VP=0, FN=2 -> F1=0
    assert m["por_categoria"]["elogio"]["f1"] == 0
    assert m["f1_macro"] == pytest.approx(0.8 / 3)
    # micro: VP=2, FP=1, FN=3 -> P=2/3, R=2/5 -> F1 = 2*(2/3*2/5)/(2/3+2/5) = 0.5
    assert m["f1_micro"] == pytest.approx(0.5)
    assert m["acerto_exato"] == pytest.approx(0.25)


def test_metricas_sentimento_invalido_conta_como_erro():
    m = metricas_sentimento(["positivo", "negativo", "neutro", "negativo"],
                            ["positivo", "negativo", None, "positivo"])
    assert m["acuracia"] == pytest.approx(0.5)


def test_concordancia_e_kappa():
    a = pd.DataFrame({"review_id": ["1", "2", "3", "4"], "sentimento": ["positivo", "negativo", "negativo", "neutro"],
                      "categorias": [["elogio"], ["atraso"], ["atraso"], ["outro"]]})
    b = pd.DataFrame({"review_id": ["1", "2", "3", "4"], "sentimento": ["positivo", "negativo", "positivo", "neutro"],
                      "categorias": [["elogio"], ["atraso"], ["outro"], ["outro"]]})
    c = concordancia(a, b, CATS)
    assert c["n"] == 4
    assert c["sentimento"]["concordancia_pct"] == pytest.approx(75)
    assert c["categorias"]["elogio"]["kappa"] == pytest.approx(1.0)
    # atraso: a=[0,1,1,0], b=[0,1,0,0] -> po=0.75, pe=0.5*0.75+0.5*0.25=0.5 -> kappa=0.5
    assert c["categorias"]["atraso"]["kappa"] == pytest.approx(0.5)
    assert c["conjunto_identico_pct"] == pytest.approx(75)


def test_checagem_atraso_ignora_desconhecidos():
    real = [1, 0, 1, None]
    prev = [["atraso"], ["atraso"], ["nao_recebido"], ["atraso"]]
    c = checagem_atraso(real, prev)
    assert c["n"] == 3
    assert c["atraso"]["cobertura_dos_atrasados_pct"] == pytest.approx(50)
    assert c["atraso"]["precisao_pct"] == pytest.approx(50)
    assert c["atraso_ou_nao_recebido"]["cobertura_dos_atrasados_pct"] == pytest.approx(100)


def test_checagem_atraso_numeros_absolutos():
    c = checagem_atraso([True, True, False, False, None], [["atraso"], [], ["atraso"], [], ["atraso"]])
    a = c["atraso"]
    assert (c["n"], a["atrasados_reais"], a["marcados"], a["marcados_e_atrasados"]) == (4, 2, 2, 1)
    assert a["cobertura_dos_atrasados_pct"] == pytest.approx(50)
    assert a["precisao_pct"] == pytest.approx(50)


def test_bootstrap_f1_macro_pontual_bate_com_sklearn_e_ic_contem():
    import numpy as np
    from metricas import bootstrap_f1_macro
    rng = np.random.default_rng(0)
    verdade = [[c for c in CATS if rng.random() < 0.4] for _ in range(200)]
    bom = [v if rng.random() < 0.8 else ["outro"] for v in verdade]
    ruim = [[c for c in CATS if rng.random() < 0.4] for _ in range(200)]
    b = bootstrap_f1_macro(verdade, {"bom": bom, "ruim": ruim}, pares=[("bom", "ruim")],
                           n_reamostras=300, seed=1, categorias=CATS)
    for nome, prev in [("bom", bom), ("ruim", ruim)]:
        r = b["modelos"][nome]
        assert r["f1_macro"] == pytest.approx(metricas_categorias(verdade, prev, CATS)["f1_macro"])
        assert r["ic95"][0] <= r["f1_macro"] <= r["ic95"][1]
    d = b["diferencas"]["bom − ruim"]
    assert d["ic95"][0] > 0 and d["fracao_reamostras_a_maior"] == 1.0
    # mesma seed -> mesmo resultado
    assert bootstrap_f1_macro(verdade, {"bom": bom}, n_reamostras=300, seed=1, categorias=CATS)["modelos"]["bom"] == b["modelos"]["bom"]
