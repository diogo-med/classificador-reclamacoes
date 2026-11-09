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
