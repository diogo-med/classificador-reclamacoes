"""Etapa 10: importa o GGUF ajustado no Ollama local e confere se as saídas batem com as do Kaggle.

1. Gera um Modelfile a partir de ollama/Modelfile (mesmo TEMPLATE do treino), trocando o FROM
   pelo GGUF indicado, e roda `ollama create`.
2. Roda os primeiros N exemplos da validação (ordem de review_id) no Ollama local, com o mesmo
   prompt de baseline, temperatura 0 e raciocínio desligado.
3. Compara com as previsões do Kaggle (transformers, 16 bits): texto idêntico, JSON idêntico
   e categorias/sentimento idênticos. Diferenças pequenas são esperadas porque o GGUF é q8_0.

Uso:
  python src/conferir_ollama.py --gguf modelos/qwen2b_lora-Q8_0.gguf --nome cr-qwen2b-lora \
      --kaggle outputs/kaggle/treino_qwen2b_lora/pred_qwen2b_lora_validacao.jsonl \
      --saida resultados/conferencia_ollama_kaggle.json
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATA_PROCESSED, OUTPUTS, RAIZ  # noqa: E402
from esquema import avaliar_resposta, extrair_json  # noqa: E402
from prompts import prompt_baseline  # noqa: E402
from rotular import chamar_ollama  # noqa: E402


def criar_modelo(gguf: Path, nome: str):
    modelo = (RAIZ / "ollama" / "Modelfile").read_text()
    linhas = [f"FROM {gguf.resolve()}" if l.startswith("FROM ") else l for l in modelo.splitlines()]
    destino = OUTPUTS / f"Modelfile_{nome}"
    destino.write_text("\n".join(linhas) + "\n")
    subprocess.run(["ollama", "create", nome, "-f", str(destino)], check=True)


def conferir(nome: str, kaggle_jsonl: Path, n: int) -> dict:
    kaggle = {json.loads(l)["review_id"]: json.loads(l)["resposta"]
              for l in kaggle_jsonl.read_text(encoding="utf-8").splitlines()}
    amostra = pd.read_parquet(DATA_PROCESSED / "amostra.parquet")
    val = amostra[amostra["split"] == "validacao"].sort_values("review_id").head(n)
    linhas = []
    for rid, texto in zip(val["review_id"], val["texto"]):
        resp, _, seg, info = chamar_ollama(prompt_baseline(texto), nome, num_predict=256, num_ctx=2048)
        a, b = avaliar_resposta(resp), avaliar_resposta(kaggle[rid])
        linhas.append({
            "review_id": rid, "ollama": resp, "kaggle": kaggle[rid],
            "texto_identico": resp.strip() == kaggle[rid].strip(),
            "json_identico": extrair_json(resp) == extrair_json(kaggle[rid]),
            "rotulos_identicos": (a["sentimento"], a["categorias"]) == (b["sentimento"], b["categorias"]),
            "prompt_tokens": info.get("prompt_eval_count"),
        })
    df = pd.DataFrame(linhas)
    return {
        "modelo_ollama": nome, "n": int(len(df)),
        "texto_identico": int(df["texto_identico"].sum()),
        "json_identico": int(df["json_identico"].sum()),
        "rotulos_identicos": int(df["rotulos_identicos"].sum()),
        "exemplos": linhas,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gguf", required=True)
    p.add_argument("--nome", required=True)
    p.add_argument("--kaggle", required=True)
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--saida", required=True)
    a = p.parse_args()
    criar_modelo(Path(a.gguf), a.nome)
    r = conferir(a.nome, Path(a.kaggle), a.n)
    Path(a.saida).parent.mkdir(parents=True, exist_ok=True)
    Path(a.saida).write_text(json.dumps(r, ensure_ascii=False, indent=2))
    print(f"{a.nome}: texto idêntico {r['texto_identico']}/{r['n']}, JSON idêntico {r['json_identico']}/{r['n']}, "
          f"rótulos idênticos {r['rotulos_identicos']}/{r['n']}")


if __name__ == "__main__":
    main()
