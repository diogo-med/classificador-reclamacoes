"""Nova tentativa e cota diária do Gemini, sem chamadas reais (cliente e erros falsos)."""
import json
from types import SimpleNamespace

import pandas as pd
import pytest
from google.genai import errors

import eficiencia
import inferir
import rotular


def erro_429(quota_id: str, atraso: str = "3s"):
    return errors.ClientError(429, {"error": {
        "code": 429, "status": "RESOURCE_EXHAUSTED",
        "message": f"You exceeded your current quota.\n* Quota exceeded for metric: x, limit: 20\n"
                   f"Please retry in {atraso}.",
        "details": [
            {"@type": "type.googleapis.com/google.rpc.QuotaFailure",
             "violations": [{"quotaMetric": "generativelanguage.googleapis.com/generate_content_free_tier_requests",
                             "quotaId": quota_id, "quotaValue": "20"}]},
            {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": atraso},
        ]}})


DIARIO = "GenerateRequestsPerDayPerProjectPerModel-FreeTier"
POR_MINUTO = "GenerateRequestsPerMinutePerProjectPerModel-FreeTier"


def erro_503():
    return errors.ServerError(503, {"error": {"code": 503, "status": "UNAVAILABLE",
                                              "message": "The model is overloaded."}})


def resposta(texto='{"avaliacoes": []}'):
    uso = SimpleNamespace(prompt_token_count=100, candidates_token_count=20, thoughts_token_count=5)
    return SimpleNamespace(text=texto, model_version="gemini-falso", usage_metadata=uso)


class ClienteFalso:
    """Devolve (ou levanta) os itens de `roteiro` em ordem, um por chamada."""
    def __init__(self, roteiro):
        self.roteiro, self.chamadas = list(roteiro), []
        self.models = SimpleNamespace(generate_content=self._gerar)

    def _gerar(self, model, contents, config):
        self.chamadas.append(config)
        item = self.roteiro.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


@pytest.fixture
def esperas(monkeypatch):
    feitas = []
    monkeypatch.setattr(rotular.time, "sleep", feitas.append)
    return feitas


def test_analisar_429_diario_e_por_minuto():
    assert rotular.analisar_429(erro_429(DIARIO, "30619s"))[0] is True
    diaria, atraso, _ = rotular.analisar_429(erro_429(POR_MINUTO, "7s"))
    assert diaria is False and atraso == 7
    # mensagem real (repr no texto do erro, sem detalhes estruturados)
    sem_det = errors.ClientError(429, {"error": {"code": 429, "message": (
        "generate_content_free_tier_requests, 'quotaId': 'GenerateRequestsPerDayPerProjectPerModel-FreeTier'")}})
    assert rotular.analisar_429(sem_det)[0] is True
    # espera sugerida enorme também conta como diária
    assert rotular.analisar_429(erro_429(POR_MINUTO, "7200s"))[0] is True


def test_429_diario_levanta_cota_sem_esperar(esperas):
    cliente = ClienteFalso([erro_429(DIARIO)])
    with pytest.raises(rotular.CotaDiariaEsgotada, match="PerDay"):
        rotular.chamar_gemini(cliente, "p", {}, "m")
    assert esperas == [] and len(cliente.chamadas) == 1


def test_429_por_minuto_espera_e_tenta_de_novo(esperas):
    cliente = ClienteFalso([erro_429(POR_MINUTO, "7s"), resposta()])
    texto, versao, _, info = rotular.chamar_gemini(cliente, "p", {"type": "object"}, "m")
    assert texto == '{"avaliacoes": []}' and versao == "gemini-falso"
    assert esperas == [9]                                     # sugerido + 2 s
    assert info == {"prompt_eval_count": 100, "eval_count": 25, "tokens_raciocinio": 5}
    assert cliente.chamadas[0].response_mime_type == "application/json"


def test_503_tenta_de_novo_com_espera_crescente(esperas):
    cliente = ClienteFalso([erro_503(), erro_503(), resposta("{}")])
    texto, *_ = inferir.chamar_gemini_livre(cliente, "p", "m")
    assert texto == "{}" and esperas == [10, 20]
    assert cliente.chamadas[0].response_mime_type is None    # chamada livre: sem esquema
    assert cliente.chamadas[0].temperature == 0


def test_erro_400_nao_tenta_de_novo(esperas):
    cliente = ClienteFalso([errors.ClientError(400, {"error": {"code": 400, "message": "ruim"}})])
    with pytest.raises(errors.ClientError):
        inferir.chamar_gemini_livre(cliente, "p", "m")
    assert esperas == []


def test_falha_depois_de_8_tentativas(esperas):
    cliente = ClienteFalso([erro_503()] * 8)
    with pytest.raises(RuntimeError, match="8 tentativas"):
        rotular.chamar_gemini(cliente, "p", {}, "m")
    assert len(esperas) == 8


def _df(n):
    return pd.DataFrame({"review_id": [f"r{i}" for i in range(n)], "texto": [f"t{i}" for i in range(n)]})


def test_inferir_para_limpo_na_cota_e_preserva_o_gravado(tmp_path, monkeypatch, esperas):
    cliente = ClienteFalso([resposta("{}"), resposta("{}"), erro_429(DIARIO)])
    monkeypatch.setattr(inferir, "_cliente_gemini", lambda: cliente)
    saida = tmp_path / "pred.jsonl"
    assert inferir.inferir(_df(4), "gemini", None, saida) == 2
    linhas = [json.loads(l) for l in saida.read_text().splitlines()]
    assert [l["review_id"] for l in linhas] == ["r0", "r1"] and linhas[0]["eval_count"] == 25
    # retoma de onde parou
    cliente.roteiro = [resposta("{}"), resposta("{}")]
    assert inferir.inferir(_df(4), "gemini", None, saida) == 2
    assert len(saida.read_text().splitlines()) == 4


def test_rotular_grava_tokens_e_para_na_cota(tmp_path, monkeypatch, esperas):
    df = _df(3)
    # a resposta traz os 3; validar_lote só aceita os ids pedidos no lote
    lote = json.dumps({"avaliacoes": [{"id": f"r{i}", "sentimento": "positivo", "categorias": ["elogio"],
                                       "resumo": "ok"} for i in range(3)]})
    cliente = ClienteFalso([resposta(lote), erro_429(DIARIO)])
    monkeypatch.setattr(rotular, "_cliente_gemini", lambda: cliente)
    saida = tmp_path / "rot.jsonl"
    rotular.rotular(df, "gemini", saida, tamanho_lote=2)   # 2 lotes; o 2º esbarra na cota
    regs = [json.loads(l) for l in saida.read_text().splitlines()]
    assert len(regs) == 1 and len(regs[0]["rotulos"]) == 2
    assert regs[0]["prompt_eval_count"] == 100 and regs[0]["eval_count"] == 25
    assert regs[0]["tokens_raciocinio"] == 5
    assert set(rotular.ler_progresso(saida)) == set(regs[0]["lote_ids"])
    assert len(rotular.consolidar(saida)) == 2


def _preparar_eficiencia(tmp_path, monkeypatch, roteiro):
    cliente = ClienteFalso(roteiro)
    monkeypatch.setattr(inferir, "_cliente_gemini", lambda: cliente)
    monkeypatch.setattr(inferir.time, "sleep", lambda s: None)
    monkeypatch.setattr(eficiencia, "PASTA", tmp_path)
    monkeypatch.setattr(eficiencia, "ids_fixos", lambda: [f"r{i}" for i in range(5)])
    monkeypatch.setattr(eficiencia, "selecionar", lambda *a: _df(5))


def test_medir_gemini_registra_n_real_quando_a_cota_acaba(tmp_path, monkeypatch, esperas):
    _preparar_eficiencia(tmp_path, monkeypatch, [resposta("{}"), resposta("{}"), erro_429(DIARIO)])
    eficiencia.medir_gemini("Gemini teste", 5)
    reg = json.loads((tmp_path / "registro.json").read_text())
    assert "Gemini teste" in reg
    assert eficiencia.resumir(tmp_path / reg["Gemini teste"]["arquivo"], "API")["n"] == 2


def test_medir_gemini_nao_registra_sem_nenhuma_chamada(tmp_path, monkeypatch, esperas):
    _preparar_eficiencia(tmp_path, monkeypatch, [erro_429(DIARIO)])
    eficiencia.medir_gemini("Gemini teste", 5)
    assert not (tmp_path / "registro.json").exists()
