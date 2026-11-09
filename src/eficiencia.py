"""Etapa 11b: eficiência na mesma CPU (Ollama local) e latência da API do Gemini.

- Amostra fixa de 100 avaliações do teste (seed 42), salva em resultados/ids_eficiencia.txt.
- Uma chamada de aquecimento por modelo (carrega o modelo na memória) não entra na conta.
- Latência = tempo de parede da chamada HTTP; tokens/s = eval_count / eval_duration do Ollama.
- Custo por 1.000 avaliações: só onde há preço oficial registrado em PRECOS (com fonte e data).

Uso:
  python src/eficiencia.py --modelo qwen3.5:2b --nome "Qwen3.5-2B sem ajuste (q8_0)"
  python src/eficiencia.py --modelo cr-qwen2b-lora --nome "Qwen3.5-2B + LoRA (q8_0)"
  python src/eficiencia.py --modelo cr-qwen2b-lora --nome "Qwen3.5-2B + LoRA (q8_0, format=esquema)" --restrito
  python src/eficiencia.py --gemini --n 10 --nome "Gemini 3.6 Flash (API)"
  python src/eficiencia.py --so-tabela        # só recalcula resultados/eficiencia.json
"""
import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import OUTPUTS, RESULTADOS, SEED  # noqa: E402
from esquema import avaliar_resposta  # noqa: E402
from inferir import inferir, selecionar  # noqa: E402
from rotular import chamar_ollama  # noqa: E402

N = 100
PASTA = RESULTADOS / "eficiencia"
IDS = RESULTADOS / "ids_eficiencia.txt"

# Preços oficiais consultados (preencher só com fonte e data). Valores em US$ por 1M de tokens.
PRECOS = json.loads((RESULTADOS / "precos.json").read_text()) if (RESULTADOS / "precos.json").exists() else {}


def ids_fixos() -> list[str]:
    if not IDS.exists():
        df = selecionar("teste", N, None)
        IDS.write_text("\n".join(df["review_id"]) + "\n")
    return IDS.read_text().split()


def hardware() -> dict:
    lscpu = subprocess.run(["lscpu"], capture_output=True, text=True, env={"LC_ALL": "C"}).stdout
    mem = subprocess.run(["free", "-h"], capture_output=True, text=True, env={"LC_ALL": "C"}).stdout
    campos = dict(l.split(":", 1) for l in lscpu.splitlines() if ":" in l)
    nome = campos.get("Model name", "?").strip()
    cpus = campos.get("CPU(s)", "?").strip()
    ram = mem.splitlines()[1].split()[1] if len(mem.splitlines()) > 1 else "?"
    (RESULTADOS / "hardware.txt").write_text(f"{platform.platform()}\n\n$ lscpu\n{lscpu}\n$ free -h\n{mem}")
    return {"resumo": f"{nome}, {cpus} CPUs lógicas, {ram} de RAM, sem GPU dedicada (só CPU)",
            "cpu": nome, "cpus": cpus, "ram": ram}


def medir_ollama(modelo: str, nome: str, restrito: bool):
    df = selecionar("teste", None, None).set_index("review_id").loc[ids_fixos()].reset_index()
    chamar_ollama("Responda só: ok", modelo, num_predict=8, num_ctx=2048)  # aquecimento
    saida = PASTA / f"{arquivo(nome)}.jsonl"
    inferir(df, "ollama", modelo, saida, restrito=restrito)
    registrar(nome, saida, onde="Ollama local (CPU)")


def medir_gemini(nome: str, n: int):
    df = selecionar("teste", None, None).set_index("review_id").loc[ids_fixos()[:n]].reset_index()
    saida = PASTA / f"{arquivo(nome)}.jsonl"
    inferir(df, "gemini", None, saida, pausa=4)
    registrar(nome, saida, onde="API (rede)")


def arquivo(nome: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in nome).strip("_").lower()


def registrar(nome: str, caminho: Path, onde: str):
    meta = PASTA / "registro.json"
    reg = json.loads(meta.read_text()) if meta.exists() else {}
    reg[nome] = {"arquivo": caminho.name, "onde": onde}
    meta.write_text(json.dumps(reg, ensure_ascii=False, indent=2))


def resumir(caminho: Path, onde: str) -> dict:
    linhas = [json.loads(l) for l in caminho.read_text(encoding="utf-8").splitlines()]
    seg = np.array([l["segundos"] for l in linhas])
    av = [avaliar_resposta(l["resposta"]) for l in linhas]
    r = {"onde": onde, "n": len(linhas), "latencia_media_s": float(seg.mean()),
         "latencia_p95_s": float(np.percentile(seg, 95)),
         "json_valido_pct": 100 * float(np.mean([a["json_valido"] for a in av])),
         "json_no_esquema_pct": 100 * float(np.mean([a["no_esquema"] for a in av]))}
    if all(l.get("eval_count") is not None for l in linhas):
        r["tokens_saida_medio"] = float(np.mean([l["eval_count"] for l in linhas]))
        r["tokens_entrada_medio"] = float(np.mean([l["prompt_eval_count"] for l in linhas]))
    if all(l.get("eval_count") and l.get("eval_duration") for l in linhas):
        r["tokens_por_segundo"] = float(sum(l["eval_count"] for l in linhas) /
                                        (sum(l["eval_duration"] for l in linhas) / 1e9))
    return r


def tabela():
    PASTA.mkdir(parents=True, exist_ok=True)
    meta = PASTA / "registro.json"
    reg = json.loads(meta.read_text()) if meta.exists() else {}
    modelos = {nome: resumir(PASTA / d["arquivo"], d["onde"]) for nome, d in reg.items()
               if (PASTA / d["arquivo"]).exists()}
    premissas = [
        f"Mesma amostra fixa de {N} avaliações do teste para os modelos locais "
        f"(`resultados/ids_eficiencia.txt`), uma chamada por avaliação, sem paralelismo, "
        "temperatura 0, raciocínio desligado, num_ctx 2048, 1 chamada de aquecimento descartada.",
        "Latência = tempo de parede da requisição HTTP ao Ollama local (inclui processar o prompt e gerar a resposta).",
        "Modelos locais: custo marginal de API = 0; o custo real é o hardware/energia, que não estimamos.",
    ]
    for nome, r in modelos.items():
        if "Gemini" in nome:
            p = PRECOS.get("gemini-3.6-flash")
            if p and r.get("tokens_entrada_medio"):
                custo = 1000 * (r["tokens_entrada_medio"] * p["entrada_usd_por_1m"] +
                                r["tokens_saida_medio"] * p["saida_usd_por_1m"]) / 1e6
                r["custo_por_1000"] = f"US$ {custo:.3f}"
            elif p:
                r["custo_por_1000"] = "ver premissas"
            else:
                r["custo_por_1000"] = "sem preço oficial registrado"
            premissas.append(f"{nome}: latência da API medida em {r['n']} chamadas (nível gratuito, limitado "
                             "a 20 requisições/dia), inclui rede; não é comparável diretamente à CPU local.")
        else:
            r["custo_por_1000"] = "US$ 0 de API (hardware próprio)"
    if PRECOS:
        for m, p in PRECOS.items():
            premissas.append(f"Preço {m}: US$ {p['entrada_usd_por_1m']}/1M tokens de entrada e "
                             f"US$ {p['saida_usd_por_1m']}/1M de saída ({p['fonte']}, consultado em {p['data']}). "
                             + p.get("nota", ""))
    out = {"hardware": hardware(), "modelos": modelos, "premissas": premissas}
    (RESULTADOS / "eficiencia.json").write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(json.dumps(modelos, ensure_ascii=False, indent=1))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--modelo")
    p.add_argument("--nome")
    p.add_argument("--restrito", action="store_true")
    p.add_argument("--gemini", action="store_true")
    p.add_argument("--n", type=int, default=N)
    p.add_argument("--so-tabela", action="store_true")
    a = p.parse_args()
    PASTA.mkdir(parents=True, exist_ok=True)
    if not a.so_tabela:
        if a.gemini:
            medir_gemini(a.nome, a.n)
        else:
            medir_ollama(a.modelo, a.nome, a.restrito)
    tabela()


if __name__ == "__main__":
    np.random.seed(SEED)
    main()
