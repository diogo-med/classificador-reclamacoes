"""Etapas 5 e 6: rotulagem em lote com o PROMPT_ROTULAGEM.

Rotuladores:
- gemini: API do Gemini (SDK google-genai). Chave lida do .env via python-dotenv.
- ollama: qualquer modelo do Ollama (usado para o Qwen3.5-9B, no Kaggle).

O progresso é salvo em um .jsonl (uma linha por lote). Ao rodar de novo,
lotes já concluídos são pulados, então nenhuma requisição é repetida.

Uso:
  python src/rotular.py --rotulador gemini --saida outputs/rotulos_gemini.jsonl
  python src/rotular.py --rotulador ollama --modelo qwen3.5:9b \
      --ids outputs/ids_9b.txt --saida outputs/rotulos_9b.jsonl
"""
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATA_PROCESSED, SEED  # noqa: E402
from esquema import ESQUEMA_LOTE, dentro_do_esquema  # noqa: E402
from prompts import VERSAO_ROTULAGEM, prompt_rotulagem  # noqa: E402

MODELO_GEMINI = "gemini-3.6-flash"


class CotaDiariaEsgotada(Exception):
    pass


PRIORIDADE_SPLIT = {"teste": 0, "validacao": 1, "treino": 2}


def fazer_lotes(df: pd.DataFrame, tamanho: int, seed: int = SEED) -> list[list[tuple[str, str]]]:
    """Embaralha (seed fixa) e divide em lotes de (review_id, texto).

    Se houver coluna 'split', rotula primeiro teste, depois validação, depois
    treino (a cota diária do Gemini é pequena; ver DECISOES.md D19)."""
    df = df.sample(frac=1, random_state=seed)
    if "split" in df.columns:
        df = df.assign(_p=df["split"].map(PRIORIDADE_SPLIT)).sort_values("_p", kind="stable")
    pares = list(zip(df["review_id"], df["texto"]))
    return [pares[i:i + tamanho] for i in range(0, len(pares), tamanho)]


def validar_lote(obj, ids: list[str]) -> dict:
    """Retorna {id: rotulo} só com os itens válidos e cujo id foi pedido."""
    validos = {}
    if not isinstance(obj, dict) or not isinstance(obj.get("avaliacoes"), list):
        return validos
    pedidos = set(ids)
    for item in obj["avaliacoes"]:
        if not isinstance(item, dict) or item.get("id") not in pedidos:
            continue
        rotulo = {k: item.get(k) for k in ("sentimento", "categorias", "resumo")}
        if isinstance(rotulo["categorias"], list):  # repetição não invalida um rótulo
            rotulo["categorias"] = sorted(set(map(str, rotulo["categorias"])))
        if dentro_do_esquema(rotulo):
            validos[item["id"]] = rotulo
    return validos


# ---------------------------------------------------------------- Gemini
def _cliente_gemini():
    from dotenv import load_dotenv
    from google import genai
    from google.genai import types
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    if not os.getenv("GEMINI_API_KEY"):
        raise RuntimeError("GEMINI_API_KEY não encontrada no .env")
    return genai.Client(http_options=types.HttpOptions(timeout=180_000))


ESPERA_MAX_429 = 3600  # 429 pedindo para esperar mais que isso = cota diária, não vale esperar


def _diaria(cota: str) -> bool:
    c = cota.lower().replace("_", "").replace(" ", "")
    return "perday" in c


def analisar_429(e) -> tuple[bool, float | None, str]:
    """Lê um erro 429 da API. Retorna (cota diária?, espera sugerida em s ou None, descrição).

    Usa os detalhes estruturados (QuotaFailure.quotaId, RetryInfo.retryDelay) e, na falta
    deles, o texto do erro. Diária = todas as cotas estouradas são por dia."""
    msg = str(e)
    det = getattr(e, "details", None)
    erro = det.get("error", det) if isinstance(det, dict) else {}
    erro = erro if isinstance(erro, dict) else {}
    cotas, atraso = [], None
    for d in erro.get("details") or []:
        if not isinstance(d, dict):
            continue
        tipo = str(d.get("@type", ""))
        if tipo.endswith("QuotaFailure"):
            cotas += [v.get("quotaId") or v.get("quotaMetric") or v.get("description") or ""
                      for v in d.get("violations") or [] if isinstance(v, dict)]
        elif tipo.endswith("RetryInfo"):
            m = re.match(r"([\d.]+)s$", str(d.get("retryDelay", "")))
            atraso = float(m.group(1)) if m else atraso
    if not cotas:
        cotas = re.findall(r"'quotaId': '([^']+)'", msg) or [
            m.split("\\n")[0].strip() for m in re.findall(r"Quota exceeded for metric: ([^\n*]+)", msg)]
    if atraso is None:
        m = re.search(r"retry in ([\d.]+)s|'retryDelay': '(\d+)s'", msg)
        atraso = float(m.group(1) or m.group(2)) if m else None
    diaria = bool(cotas) and all(_diaria(c) for c in cotas)
    if not cotas and re.search(r"per day", msg, re.IGNORECASE):
        diaria = True
    if atraso is not None and atraso > ESPERA_MAX_429:
        diaria = True
    desc = "; ".join(cotas) if cotas else msg[:200]
    if atraso is not None:
        desc += f" (espera sugerida {atraso:.0f}s)"
    return diaria, atraso, desc


def chamar_com_retentativa(chamada, tentativas: int = 8):
    """Executa chamada() (uma requisição ao Gemini) com nova tentativa para 429 por minuto e 5xx.

    429 de cota diária -> CotaDiariaEsgotada (não adianta esperar). Espera inicial de 10 s,
    dobrando até 300 s (8 tentativas ~ 20 min). Retorna (resposta, segundos da tentativa que deu certo)."""
    from google.genai import errors
    espera = 10
    for tentativa in range(tentativas):
        inicio = time.time()
        try:
            return chamada(), time.time() - inicio
        except errors.APIError as e:
            if e.code == 429:
                diaria, sugerida, desc = analisar_429(e)
                print(f"  429: {desc}", flush=True)
                if diaria:
                    raise CotaDiariaEsgotada(desc) from e
                atraso = sugerida + 2 if sugerida is not None else espera
            elif e.code is not None and 500 <= e.code < 600:
                atraso = espera
            else:
                raise
            print(f"  erro {e.code}, aguardando {atraso:.0f}s (tentativa {tentativa + 1})", flush=True)
            time.sleep(atraso)
            espera = min(espera * 2, 300)
    raise RuntimeError(f"Falhou depois de {tentativas} tentativas")


def uso_gemini(r) -> dict:
    """Tokens de uso da resposta. eval_count = resposta + raciocínio (tudo cobrado como saída)."""
    u = getattr(r, "usage_metadata", None)
    if u is None:
        return {}
    return {"prompt_eval_count": u.prompt_token_count,
            "eval_count": (u.candidates_token_count or 0) + (u.thoughts_token_count or 0),
            "tokens_raciocinio": u.thoughts_token_count or 0}


def chamar_gemini(cliente, prompt: str, esquema: dict, modelo: str = MODELO_GEMINI):
    """Rotulagem com saída JSON no esquema. Retorna (texto, versao_modelo, segundos, info de tokens)."""
    from google.genai import types
    config = types.GenerateContentConfig(
        temperature=0, response_mime_type="application/json",
        response_json_schema=esquema)
    r, seg = chamar_com_retentativa(
        lambda: cliente.models.generate_content(model=modelo, contents=prompt, config=config))
    return r.text, r.model_version, seg, uso_gemini(r)


# ---------------------------------------------------------------- Ollama
def chamar_ollama(prompt: str, modelo: str, esquema: dict | None = None,
                  url: str = "http://localhost:11434", num_predict: int = 4096,
                  num_ctx: int = 8192):
    """Chamada ao /api/chat do Ollama com raciocínio desligado e temperatura 0.
    Retorna (texto, modelo, segundos, info) onde info tem contagem de tokens."""
    corpo = {
        "model": modelo,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "think": False,
        # presence_penalty=0: o Modelfile do qwen3.5 traz 1.5 por padrão, o que muda a
        # saída mesmo com temperatura 0; zeramos para todos os modelos serem comparáveis.
        "options": {"temperature": 0, "presence_penalty": 0, "seed": SEED,
                    "num_predict": num_predict, "num_ctx": num_ctx},
    }
    if esquema is not None:
        corpo["format"] = esquema
    inicio = time.time()
    r = requests.post(f"{url}/api/chat", json=corpo, timeout=1800)
    r.raise_for_status()
    d = r.json()
    info = {k: d.get(k) for k in ("prompt_eval_count", "eval_count", "eval_duration",
                                  "total_duration", "load_duration")}
    return d["message"]["content"], modelo, time.time() - inicio, info


# ---------------------------------------------------------------- loop
def ler_progresso(caminho: Path) -> dict:
    feitos = {}
    if caminho.exists():
        for linha in caminho.read_text(encoding="utf-8").splitlines():
            reg = json.loads(linha)
            feitos.update(reg["rotulos"])
    return feitos


def rotular(df, rotulador, saida: Path, tamanho_lote=25, modelo=None,
            url="http://localhost:11434", max_lotes=None, pausa=0.0):
    saida.parent.mkdir(parents=True, exist_ok=True)
    feitos = ler_progresso(saida)
    pendentes = df[~df["review_id"].isin(feitos)]
    lotes = fazer_lotes(pendentes, tamanho_lote)
    if max_lotes:
        lotes = lotes[:max_lotes]
    print(f"{len(feitos)} já rotuladas, {len(pendentes)} pendentes em {len(lotes)} lotes", flush=True)
    cliente = _cliente_gemini() if rotulador == "gemini" else None
    for n, lote in enumerate(lotes, 1):
        ids = [i for i, _ in lote]
        prompt = prompt_rotulagem(lote)
        try:
            if rotulador == "gemini":
                texto, versao, seg, info = chamar_gemini(cliente, prompt, ESQUEMA_LOTE, modelo or MODELO_GEMINI)
            else:
                texto, versao, seg, info = chamar_ollama(prompt, modelo, ESQUEMA_LOTE, url)
        except CotaDiariaEsgotada as e:
            print(f"Cota diária esgotada: {e}", flush=True)
            break
        try:
            obj = json.loads(texto)
        except (json.JSONDecodeError, TypeError):
            obj = None
        rotulos = validar_lote(obj, ids)
        registro = {"lote_ids": ids, "rotulos": rotulos, "modelo": versao,
                    "prompt": VERSAO_ROTULAGEM, "tamanho_lote": len(ids), "segundos": round(seg, 2),
                    "invalidos": [i for i in ids if i not in rotulos], **info}
        with open(saida, "a", encoding="utf-8") as f:
            f.write(json.dumps(registro, ensure_ascii=False) + "\n")
        print(f"lote {n}/{len(lotes)}: {len(rotulos)}/{len(ids)} válidos em {seg:.1f}s ({versao})", flush=True)
        if pausa:
            time.sleep(pausa)


def consolidar(caminho: Path) -> pd.DataFrame:
    """Lê o .jsonl de progresso e devolve um DataFrame review_id, sentimento, categorias, resumo."""
    linhas = []
    for linha in caminho.read_text(encoding="utf-8").splitlines():
        reg = json.loads(linha)
        for rid, r in reg["rotulos"].items():
            linhas.append({"review_id": rid, **r, "modelo": reg["modelo"],
                           "tamanho_lote": reg.get("tamanho_lote", len(reg["lote_ids"]))})
    return pd.DataFrame(linhas).drop_duplicates("review_id", keep="last")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rotulador", choices=["gemini", "ollama"], required=True)
    p.add_argument("--modelo", default=None)
    p.add_argument("--amostra", default=str(DATA_PROCESSED / "amostra.parquet"))
    p.add_argument("--ids", default=None, help="arquivo com um review_id por linha (opcional)")
    p.add_argument("--saida", required=True)
    p.add_argument("--tamanho-lote", type=int, default=25)
    p.add_argument("--max-lotes", type=int, default=None)
    p.add_argument("--pausa", type=float, default=0.0, help="segundos entre lotes")
    p.add_argument("--url", default="http://localhost:11434")
    a = p.parse_args()
    df = pd.read_parquet(a.amostra)
    if a.ids:
        ids = set(Path(a.ids).read_text().split())
        df = df[df["review_id"].isin(ids)]
    rotular(df, a.rotulador, Path(a.saida), a.tamanho_lote, a.modelo, a.url, a.max_lotes, a.pausa)


if __name__ == "__main__":
    main()
