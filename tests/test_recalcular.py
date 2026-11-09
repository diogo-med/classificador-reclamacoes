import json

import pandas as pd
import pytest

import recalcular


@pytest.fixture
def pasta(tmp_path):
    gemini = pd.DataFrame({
        "review_id": ["t1", "t2", "a1", "d1", "x9"],
        "sentimento": ["negativo", "positivo", "negativo", "neutro", "positivo"],
        "categorias": [["atraso"], ["elogio"], ["defeito_qualidade"], ["outro"], ["elogio"]],
        "resumo": ["r"] * 5, "modelo": ["gemini-3.6-flash"] * 5,
    })
    gemini.to_parquet(tmp_path / "rotulos_gemini.parquet", index=False)
    planilha = pd.DataFrame({
        "review_id": ["t1", "t2", "a1", "d1"],
        "bloco": ["teste", "teste", "treino_aleatorio", "treino_discordancia"],
        "nota": ["1", "5", "2", "3"], "texto": ["..."] * 4,
        "sentimento_sugerido": ["negativo", "positivo", "negativo", "neutro"],
        "categorias_sugeridas": ["atraso", "elogio", "defeito_qualidade", "outro"],
        # t1: aceita; t2: corrige categorias; a1: corrige sentimento; d1: corrige ambos
        "sentimento_revisado": ["", "", "neutro", "negativo"],
        "categorias_revisadas": ["", "elogio;atraso", "", "atendimento"],
        "observacao": [""] * 4,
    })
    csv = tmp_path / "revisao.csv"
    planilha.to_csv(csv, index=False)
    return tmp_path, csv


def test_recalcular_aplica_revisao_e_conta_mudancas(pasta):
    tmp, csv = pasta
    rel = recalcular.recalcular(csv, destino_dir=tmp, resultados_dir=tmp / "res")
    final = pd.read_parquet(tmp / "rotulos_revisados.parquet").set_index("review_id")
    assert list(final.loc["t1", "categorias"]) == ["atraso"]                 # vazio = aceito
    assert sorted(final.loc["t2", "categorias"]) == ["atraso", "elogio"]
    assert final.loc["a1", "sentimento"] == "neutro"
    assert list(final.loc["d1", "categorias"]) == ["atendimento"]
    assert list(final.loc["x9", "categorias"]) == ["elogio"]                 # fora da planilha: mantém
    assert bool(final.loc["x9", "revisado_por_humano"]) is False
    assert rel["linhas_com_correcao"] == 3
    assert rel["rotulos_de_treino_alterados"] == 2
    assert "comandos_retreino" in rel
    c = rel["concordancia_gemini_humano"]
    assert c["teste"]["n"] == 2 and c["teste"]["conjunto_identico_pct"] == pytest.approx(50)
    assert c["treino_aleatorio"]["n"] == 1
    assert "aviso" in c["treino_discordancia"]
    assert json.loads((tmp / "res" / "revisao_humana.json").read_text())["linhas_revisadas"] == 4


def test_recalcular_rejeita_valor_invalido(pasta):
    tmp, csv = pasta
    df = pd.read_csv(csv, dtype=str, keep_default_na=False)
    df.loc[0, "categorias_revisadas"] = "frete"
    df.to_csv(csv, index=False)
    with pytest.raises(ValueError, match="frete"):
        recalcular.recalcular(csv, destino_dir=tmp, resultados_dir=tmp / "res")
