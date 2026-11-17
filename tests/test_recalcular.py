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
        "conferido": ["x", "X", "x", "x"],
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
    salvo = json.loads((tmp / "res" / "revisao_humana.json").read_text())
    assert salvo["linhas_revisadas"] == 4
    assert salvo["conferidas_por_bloco"] == {"teste": {"conferidas": 2, "total": 2},
                                              "treino_aleatorio": {"conferidas": 1, "total": 1},
                                              "treino_discordancia": {"conferidas": 1, "total": 1}}


def test_recalcular_rejeita_valor_invalido(pasta):
    tmp, csv = pasta
    df = pd.read_csv(csv, dtype=str, keep_default_na=False)
    df.loc[0, "categorias_revisadas"] = "frete"
    df.to_csv(csv, index=False)
    with pytest.raises(ValueError, match="frete"):
        recalcular.recalcular(csv, destino_dir=tmp, resultados_dir=tmp / "res")


def _editar(csv, **colunas):
    df = pd.read_csv(csv, dtype=str, keep_default_na=False)
    for col, valores in colunas.items():
        df[col] = valores
    df.to_csv(csv, index=False)


def test_recalcular_recusa_teste_sem_x_e_diz_quantas_faltam(pasta):
    tmp, csv = pasta
    _editar(csv, conferido=["", "", "x", "x"])
    with pytest.raises(recalcular.RevisaoIncompleta, match="Faltam 2 de 2 linhas do bloco 'teste'"):
        recalcular.recalcular(csv, destino_dir=tmp, resultados_dir=tmp / "res")
    assert not (tmp / "rotulos_revisados.parquet").exists()   # nada é gravado


def test_recalcular_recusa_planilha_sem_coluna_conferido(pasta):
    tmp, csv = pasta
    df = pd.read_csv(csv, dtype=str, keep_default_na=False).drop(columns="conferido")
    df.to_csv(csv, index=False)
    with pytest.raises(ValueError, match="conferido"):
        recalcular.recalcular(csv, destino_dir=tmp, resultados_dir=tmp / "res")


def test_recalcular_ignora_treino_sem_x(pasta):
    tmp, csv = pasta
    # a1 e d1 sem x e sem correção: ficam com o rótulo do Gemini e fora da concordância
    _editar(csv, conferido=["x", "x", "", ""], sentimento_revisado=["", "", "", ""],
            categorias_revisadas=["", "elogio;atraso", "", ""])
    rel = recalcular.recalcular(csv, destino_dir=tmp, resultados_dir=tmp / "res")
    assert rel["conferidas_por_bloco"]["treino_aleatorio"] == {"conferidas": 0, "total": 1}
    assert rel["conferidas_por_bloco"]["treino_discordancia"] == {"conferidas": 0, "total": 1}
    assert set(rel["concordancia_gemini_humano"]) == {"teste"}
    final = pd.read_parquet(tmp / "rotulos_revisados.parquet").set_index("review_id")
    assert bool(final.loc["a1", "revisado_por_humano"]) is False
    assert final.loc["a1", "sentimento"] == "negativo"
    assert rel["rotulos_de_treino_alterados"] == 0


def test_recalcular_recusa_correcao_sem_x(pasta):
    tmp, csv = pasta
    _editar(csv, conferido=["x", "x", "", "x"])   # a1 tem sentimento corrigido mas não tem x
    with pytest.raises(ValueError, match="a1.*não tem x"):
        recalcular.recalcular(csv, destino_dir=tmp, resultados_dir=tmp / "res")


def test_recalcular_recusa_marca_invalida(pasta):
    tmp, csv = pasta
    _editar(csv, conferido=["x", "sim", "x", "x"])
    with pytest.raises(ValueError, match="deve ser x ou vazio"):
        recalcular.recalcular(csv, destino_dir=tmp, resultados_dir=tmp / "res")


def test_erro_de_valor_aponta_linha_do_csv_original(pasta):
    tmp, csv = pasta
    # linha 3 do CSV (t2) sem x seria recusada; aqui todas têm x e d1 (linha 5) tem valor inválido
    _editar(csv, categorias_revisadas=["", "elogio;atraso", "", "frete"])
    with pytest.raises(ValueError, match="linha 5 \\(d1\\)"):
        recalcular.recalcular(csv, destino_dir=tmp, resultados_dir=tmp / "res")


def test_revisao_gera_coluna_conferido_vazia():
    import revisao
    amostra = pd.DataFrame({"review_id": ["t1", "a1"], "split": ["teste", "treino"], "nota": [1, 5],
                            "texto": ["a", "b"]})
    gemini = pd.DataFrame({"review_id": ["t1", "a1"], "sentimento": ["negativo", "positivo"],
                           "categorias": [["atraso"], ["elogio"]]})
    plan = revisao.montar_planilha(amostra, gemini, None)
    assert "conferido" in plan.columns and (plan["conferido"] == "").all()
    assert plan.columns.get_loc("conferido") < plan.columns.get_loc("sentimento_revisado")


def test_wilson_casos_conhecidos():
    assert recalcular.wilson(0, 0) is None
    lo, hi = recalcular.wilson(0, 10)
    assert lo == 0 and hi == pytest.approx(0.2775, abs=1e-4)
    assert recalcular.wilson(5, 10) == pytest.approx((0.2366, 0.7634), abs=1e-4)
    assert recalcular.wilson(10, 100) == pytest.approx((0.0552, 0.1744), abs=1e-4)
    lo, hi = recalcular.wilson(10, 10)
    assert lo == pytest.approx(0.7225, abs=1e-4) and hi == pytest.approx(1)


def _pasta_aleatorio(tmp_path, n, alteradas, n_discordancia=0):
    """n linhas conferidas no treino_aleatorio (as `alteradas` primeiras com sentimento corrigido),
    1 do teste e n_discordancia do bloco de discordância, todas corrigidas."""
    ids = ["t1"] + [f"a{i}" for i in range(n)] + [f"d{i}" for i in range(n_discordancia)]
    pd.DataFrame({"review_id": ids, "sentimento": "negativo", "categorias": [["atraso"]] * len(ids),
                  "resumo": "r", "modelo": "g"}).to_parquet(tmp_path / "rotulos_gemini.parquet", index=False)
    blocos = ["teste"] + ["treino_aleatorio"] * n + ["treino_discordancia"] * n_discordancia
    corr = [""] + ["neutro" if i < alteradas else "" for i in range(n)] + ["positivo"] * n_discordancia
    csv = tmp_path / "revisao.csv"
    pd.DataFrame({"review_id": ids, "bloco": blocos, "nota": "1", "texto": "...",
                  "sentimento_sugerido": "negativo", "categorias_sugeridas": "atraso", "conferido": "x",
                  "sentimento_revisado": corr, "categorias_revisadas": "", "observacao": ""}).to_csv(csv, index=False)
    return recalcular.recalcular(csv, destino_dir=tmp_path, resultados_dir=tmp_path / "res")


def test_retreino_opcional_abaixo_do_limiar(tmp_path):
    rel = _pasta_aleatorio(tmp_path, n=100, alteradas=9, n_discordancia=30)
    e = rel["estimativa_erro_gemini_treino"]
    assert e["decisao"] == "opcional" and rel["retreino_recomendado"] is False
    assert e["n_conferidas"] == 100 and e["taxa_alteracao"] == pytest.approx(0.09)
    assert e["ic95_wilson"] == pytest.approx(list(recalcular.wilson(9, 100)))
    assert rel["recomendacao"].startswith("Retreino opcional") and rel["comandos_retreino"]
    # discordância não entra na taxa, mas conta nos rótulos de treino alterados
    assert rel["rotulos_de_treino_alterados"] == 39
    salvo = json.loads((tmp_path / "res" / "revisao_humana.json").read_text())
    assert salvo["estimativa_erro_gemini_treino"]["limiar_taxa"] == recalcular.LIMIAR_ALTERACAO_RETREINO


def test_retreino_recomendado_no_limiar_ou_acima(tmp_path):
    rel = _pasta_aleatorio(tmp_path, n=50, alteradas=5)   # exatamente 10% com o mínimo de 50
    assert rel["estimativa_erro_gemini_treino"]["decisao"] == "retreinar"
    assert rel["retreino_recomendado"] is True
    assert rel["recomendacao"].startswith("Retreino recomendado")


def test_retreino_amostra_insuficiente(tmp_path):
    rel = _pasta_aleatorio(tmp_path, n=49, alteradas=30)  # taxa alta, mas n < 50
    e = rel["estimativa_erro_gemini_treino"]
    assert e["decisao"] == "amostra_insuficiente" and rel["retreino_recomendado"] is False
    assert "Amostra insuficiente" in rel["recomendacao"] and e["ic95_wilson"] is not None


def test_retreino_sem_linhas_conferidas_no_aleatorio(tmp_path):
    rel = _pasta_aleatorio(tmp_path, n=0, alteradas=0)
    e = rel["estimativa_erro_gemini_treino"]
    assert e["decisao"] == "amostra_insuficiente" and e["taxa_alteracao"] is None and e["ic95_wilson"] is None


def test_correcao_igual_ao_sugerido_nao_conta(pasta):
    tmp, csv = pasta
    # t1 repete o sentimento sugerido; t2 e a1 repetem as categorias sugeridas
    _editar(csv, sentimento_revisado=["negativo", "", "", "negativo"],
            categorias_revisadas=["", "elogio", "defeito_qualidade", "atendimento"])
    rel = recalcular.recalcular(csv, destino_dir=tmp, resultados_dir=tmp / "res")
    assert rel["linhas_com_correcao"] == 1                  # só d1 mudou de fato
    assert rel["rotulos_de_treino_alterados"] == 1
    assert rel["estimativa_erro_gemini_treino"]["alteradas"] == 0
