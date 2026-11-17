import numpy as np
import pytest

from avaliar import arquivo_rotulos_treino, mascara_nao_vistos, pesos_por_nota


def test_mascara_nao_vistos_usa_texto_normalizado():
    teste = ["Muito bom!", "Não recebi o produto", "muito  BOM", "chegou quebrado"]
    vistos = ["muito bom", "chegou quebrado e sem nota"]
    assert mascara_nao_vistos(teste, vistos).tolist() == [False, True, False, True]


def test_pesos_por_nota_reponderam_para_a_populacao():
    notas = [1, 1, 1, 5]  # teste: 75% nota 1, 25% nota 5
    pesos = pesos_por_nota(notas, {"1": 25, "5": 75})  # população: 25% / 75%
    assert pesos.tolist() == pytest.approx([1 / 3, 1 / 3, 1 / 3, 3.0])
    # a distribuição ponderada do teste passa a ser a da população
    assert pesos[np.array(notas) == 1].sum() / pesos.sum() == pytest.approx(0.25)


def test_arquivo_rotulos_treino_padrao_e_gemini():
    assert arquivo_rotulos_treino(None) == "rotulos_gemini.parquet"
    assert arquivo_rotulos_treino({"n_treino": 1200}) == "rotulos_gemini.parquet"
    assert arquivo_rotulos_treino({"arquivo_rotulos": "rotulos_gemini.parquet"}) == "rotulos_gemini.parquet"
    assert arquivo_rotulos_treino({"arquivo_rotulos": "rotulos_revisados.parquet"}) == "rotulos_revisados.parquet"
