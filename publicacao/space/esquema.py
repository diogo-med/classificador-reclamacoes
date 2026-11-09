"""Esquema JSON da saída e funções de validação/parsing.

Usado por todos os modelos: rotulagem (Gemini, 9B), baselines e modelo ajustado.
"""
import json
import re

from jsonschema import Draft202012Validator

from config import CATEGORIAS, SENTIMENTOS

ESQUEMA_SAIDA = {
    "type": "object",
    "properties": {
        "sentimento": {"type": "string", "enum": SENTIMENTOS},
        "categorias": {
            "type": "array",
            "items": {"type": "string", "enum": CATEGORIAS},
            "minItems": 1,
            "uniqueItems": True,
        },
        "resumo": {"type": "string"},
    },
    "required": ["sentimento", "categorias", "resumo"],
}

# Esquema da resposta em lote (rotulagem): lista de objetos com id.
ESQUEMA_LOTE = {
    "type": "object",
    "properties": {
        "avaliacoes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"id": {"type": "string"}, **ESQUEMA_SAIDA["properties"]},
                "required": ["id", "sentimento", "categorias", "resumo"],
            },
        }
    },
    "required": ["avaliacoes"],
}

_validador = Draft202012Validator(ESQUEMA_SAIDA)


def extrair_json(texto: str):
    """Tenta extrair um objeto JSON de uma resposta de modelo.

    Aceita JSON puro ou JSON dentro de ```json ... ```. Retorna o objeto ou
    None se não houver JSON válido."""
    if texto is None:
        return None
    texto = texto.strip()
    # remove bloco de raciocínio, caso algum modelo o emita mesmo desligado
    texto = re.sub(r"<think>.*?</think>", "", texto, flags=re.DOTALL).strip()
    cerca = re.search(r"```(?:json)?\s*(.*?)```", texto, flags=re.DOTALL)
    if cerca:
        texto = cerca.group(1).strip()
    try:
        return json.loads(texto)
    except (json.JSONDecodeError, TypeError):
        pass
    # último recurso: primeiro '{' até o último '}'
    ini, fim = texto.find("{"), texto.rfind("}")
    if ini != -1 and fim > ini:
        try:
            return json.loads(texto[ini:fim + 1])
        except json.JSONDecodeError:
            return None
    return None


def dentro_do_esquema(obj) -> bool:
    """True se obj segue ESQUEMA_SAIDA (sentimento e categorias da lista)."""
    return obj is not None and _validador.is_valid(obj)


def avaliar_resposta(texto: str) -> dict:
    """Converte a resposta crua de um modelo num registro padronizado.

    Retorna json_valido, no_esquema, sentimento e categorias. Se o JSON for
    válido mas fora do esquema, aproveita o que der (categorias válidas) para
    as métricas de classificação, e marca no_esquema=False."""
    obj = extrair_json(texto)
    json_valido = isinstance(obj, dict)
    no_esquema = dentro_do_esquema(obj) if json_valido else False
    sentimento, categorias = None, []
    if json_valido:
        s = obj.get("sentimento")
        sentimento = s if s in SENTIMENTOS else None
        cats = obj.get("categorias")
        if isinstance(cats, list):
            categorias = sorted({c for c in cats if c in CATEGORIAS})
    return {"json_valido": json_valido, "no_esquema": no_esquema,
            "sentimento": sentimento, "categorias": categorias}
