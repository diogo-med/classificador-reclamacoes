"""Inferência com o PROMPT_BASELINE (uma avaliação por chamada).

Backends:
- ollama: qualquer modelo do Ollama (Qwen3.5-2B, Qwen3.5-9B, modelo ajustado em GGUF).
- gemini: API do Gemini.

Salva uma linha por avaliação em .jsonl (resposta crua, latência, tokens) e
retoma de onde parou. As métricas são calculadas depois em src/avaliar.py.

Uso:
  python src/inferir.py --backend ollama --modelo qwen3.5:2b --split teste \
      --saida outputs/pred_qwen2b_teste.jsonl
  python src/inferir.py --backend ollama --modelo qwen3.5:2b --split teste --n 20 \
      --saida outputs/medicao_qwen2b.jsonl      # medição de tempo (regra das 2 h)
"""
import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATA_PROCESSED, SEED  # noqa: E402
from esquema import ESQUEMA_SAIDA  # noqa: E402
from prompts import VERSAO_BASELINE, prompt_baseline  # noqa: E402
from rotular import (MODELO_GEMINI, _cliente_gemini, chamar_gemini,  # noqa: E402
                     chamar_ollama)


def selecionar(split: str, n: int | None, ids_arquivo: str | None) -> pd.DataFrame:
    df = pd.read_parquet(DATA_PROCESSED / "amostra.parquet")
    if ids_arquivo:
        ids = Path(ids_arquivo).read_text().split()
        df = df.set_index("review_id").loc[ids].reset_index()
    elif split != "todos":
        df = df[df["split"] == split].sort_values("review_id")
    if n:
        df = df.sample(n=n, random_state=SEED).sort_values("review_id")
    return df


def chamar_gemini_livre(cliente, prompt: str, modelo: str = MODELO_GEMINI):
    """Gemini sem esquema forçado (mesmas condições dos outros baselines)."""
    from google.genai import types
    import re
    from google.genai import errors
    config = types.GenerateContentConfig(temperature=0)
    espera = 10
    for tentativa in range(8):
        inicio = time.time()
        try:
            r = cliente.models.generate_content(model=modelo, contents=prompt, config=config)
            return r.text, r.model_version, time.time() - inicio
        except errors.APIError as e:
            if e.code == 429 or e.code >= 500:
                m = re.search(r"retry in ([\d.]+)s", str(e))
                atraso = float(m.group(1)) + 2 if m else espera
                print(f"  erro {e.code}, aguardando {atraso:.0f}s", flush=True)
                time.sleep(atraso)
                espera = min(espera * 2, 300)
            else:
                raise
    raise RuntimeError("Falhou depois de 8 tentativas")


def inferir(df, backend, modelo, saida: Path, url="http://localhost:11434",
            restrito=False, pausa=0.0):
    saida.parent.mkdir(parents=True, exist_ok=True)
    feitos = set()
    if saida.exists():
        feitos = {json.loads(l)["review_id"] for l in saida.read_text(encoding="utf-8").splitlines()}
    pendentes = df[~df["review_id"].isin(feitos)]
    print(f"{len(feitos)} feitas, {len(pendentes)} pendentes", flush=True)
    cliente = _cliente_gemini() if backend == "gemini" else None
    for n, (rid, texto) in enumerate(zip(pendentes["review_id"], pendentes["texto"]), 1):
        prompt = prompt_baseline(texto)
        info = {}
        if backend == "ollama":
            resposta, versao, seg, info = chamar_ollama(
                prompt, modelo, ESQUEMA_SAIDA if restrito else None, url, num_predict=512, num_ctx=2048)
        else:
            resposta, versao, seg = chamar_gemini_livre(cliente, prompt, modelo or MODELO_GEMINI)
        reg = {"review_id": rid, "resposta": resposta, "modelo": versao,
               "prompt": VERSAO_BASELINE, "restrito": restrito, "segundos": round(seg, 3), **info}
        with open(saida, "a", encoding="utf-8") as f:
            f.write(json.dumps(reg, ensure_ascii=False) + "\n")
        if n % 10 == 0 or n == len(pendentes):
            print(f"{n}/{len(pendentes)} ({seg:.1f}s última)", flush=True)
        if pausa:
            time.sleep(pausa)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=["ollama", "gemini"], required=True)
    p.add_argument("--modelo", default=None)
    p.add_argument("--split", default="teste")
    p.add_argument("--n", type=int, default=None)
    p.add_argument("--ids", default=None)
    p.add_argument("--saida", required=True)
    p.add_argument("--url", default="http://localhost:11434")
    p.add_argument("--restrito", action="store_true", help="decodificação restrita ao esquema (format)")
    p.add_argument("--pausa", type=float, default=0.0)
    a = p.parse_args()
    df = selecionar(a.split, a.n, a.ids)
    inferir(df, a.backend, a.modelo, Path(a.saida), a.url, a.restrito, a.pausa)


if __name__ == "__main__":
    main()
