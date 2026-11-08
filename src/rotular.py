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


def fazer_lotes(df: pd.DataFrame, tamanho: int, seed: int = SEED) -> list[list[tuple[str, str]]]:
    """Embaralha (seed fixa) e divide em lotes de (review_id, texto)."""
    df = df.sample(frac=1, random_state=seed)
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
        if dentro_do_esquema(rotulo):
            rotulo["categorias"] = sorted(set(rotulo["categorias"]))
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


def chamar_gemini(cliente, prompt: str, esquema: dict, modelo: str = MODELO_GEMINI):
    """Uma chamada com retentativa para 429/5xx. Retorna (texto, versao_modelo, segundos)."""
    from google.genai import errors, types
    config = types.GenerateContentConfig(
        temperature=0, response_mime_type="application/json",
        response_json_schema=esquema)
    espera = 10
    for tentativa in range(8):
        inicio = time.time()
        try:
            r = cliente.models.generate_content(model=modelo, contents=prompt, config=config)
            return r.text, r.model_version, time.time() - inicio
        except errors.APIError as e:
            msg = str(e)
            if e.code == 429:
                metricas = re.findall(r"Quota exceeded for metric: ([^\n*]+)", msg)
                print(f"  429: {metricas or msg[:200]}", flush=True)
                # só para se a métrica estourada for diária; limite por minuto -> espera
                if metricas and all("PerDay" in m or "per_day" in m.lower() for m in metricas):
                    raise CotaDiariaEsgotada("; ".join(metricas)) from e
                m = re.search(r"retry in ([\d.]+)s|'retryDelay': '(\d+)s'", msg)
                atraso = float(m.group(1) or m.group(2)) + 2 if m else espera
            elif e.code in (500, 502, 503, 504):
                atraso = espera
            else:
                raise
            print(f"  erro {e.code}, aguardando {atraso:.0f}s (tentativa {tentativa + 1})", flush=True)
            time.sleep(atraso)
            espera = min(espera * 2, 300)
    raise RuntimeError("Falhou depois de 8 tentativas")


# ---------------------------------------------------------------- Ollama
def chamar_ollama(prompt: str, modelo: str, esquema: dict | None = None,
                  url: str = "http://localhost:11434", num_predict: int = 4096):
    """Chamada ao /api/chat do Ollama com raciocínio desligado e temperatura 0.
    Retorna (texto, modelo, segundos, info) onde info tem contagem de tokens."""
    corpo = {
        "model": modelo,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "think": False,
        "options": {"temperature": 0, "seed": SEED, "num_predict": num_predict},
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
                texto, versao, seg = chamar_gemini(cliente, prompt, ESQUEMA_LOTE, modelo or MODELO_GEMINI)
                info = {}
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
                    "prompt": VERSAO_ROTULAGEM, "segundos": round(seg, 2),
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
            linhas.append({"review_id": rid, **r, "modelo": reg["modelo"]})
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
