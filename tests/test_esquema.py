from esquema import avaliar_resposta, dentro_do_esquema, extrair_json
from rotular import validar_lote


def test_json_valido_no_esquema():
    r = avaliar_resposta('{"sentimento": "negativo", "categorias": ["atraso"], "resumo": "x"}')
    assert r == {"json_valido": True, "no_esquema": True, "sentimento": "negativo", "categorias": ["atraso"]}


def test_json_em_cerca_markdown_e_think_vazio():
    texto = '<think>\n\n</think>\n\n```json\n{"sentimento": "positivo", "categorias": ["elogio"], "resumo": "ok"}\n```'
    assert avaliar_resposta(texto)["no_esquema"] is True


def test_json_valido_fora_do_esquema_aproveita_categorias_validas():
    r = avaliar_resposta('{"sentimento": "bravo", "categorias": ["atraso", "frete"], "resumo": "x"}')
    assert r["json_valido"] and not r["no_esquema"]
    assert r["sentimento"] is None and r["categorias"] == ["atraso"]


def test_json_quebrado():
    r = avaliar_resposta('{"sentimento": "negativo", "categorias": [')
    assert r == {"json_valido": False, "no_esquema": False, "sentimento": None, "categorias": []}


def test_esquema_exige_ao_menos_uma_categoria_e_resumo():
    assert not dentro_do_esquema({"sentimento": "neutro", "categorias": [], "resumo": "x"})
    assert not dentro_do_esquema({"sentimento": "neutro", "categorias": ["outro"]})
    assert extrair_json("nada aqui") is None


def test_validar_lote_ignora_ids_nao_pedidos_e_itens_invalidos():
    obj = {"avaliacoes": [
        {"id": "a", "sentimento": "negativo", "categorias": ["atraso", "atraso"], "resumo": "x"},
        {"id": "b", "sentimento": "negativo", "categorias": ["inventada"], "resumo": "x"},
        {"id": "zzz", "sentimento": "positivo", "categorias": ["elogio"], "resumo": "x"},
    ]}
    validos = validar_lote(obj, ["a", "b"])
    assert list(validos) == ["a"]
