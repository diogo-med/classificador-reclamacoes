"""Gera os notebooks .ipynb enviados ao Kaggle a partir de fontes .py (jupytext percent).

- kaggle/rotulador_9b/notebook.py -> notebook.ipynb (insere kaggle/comum_ollama.py)
- kaggle/treino/notebook.py -> kaggle/treino_<nome>/notebook.ipynb + kernel-metadata.json,
  um por modelo base em TREINOS.

Uso: python kaggle/gerar_notebooks.py
"""
import json
from pathlib import Path

import jupytext

PASTA = Path(__file__).resolve().parent
COMUM = (PASTA / "comum_ollama.py").read_text()
DATASETS = ["olistbr/brazilian-ecommerce", "diogomede/classificador-reclamacoes-dados"]

# nome -> (modelo base, quantizações GGUF)
TREINOS = {
    "qwen08b_lora": ("unsloth/Qwen3.5-0.8B", ["q8_0"]),          # valida o pipeline
    "qwen2b_lora": ("unsloth/Qwen3.5-2B", ["q8_0", "q4_k_m"]),   # versão principal
}


def escrever(texto: str, destino: Path):
    nb = jupytext.reads(texto, fmt="py:percent")
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    jupytext.write(nb, destino)
    print("gerado", destino)


def metadata(slug: str) -> dict:
    return {"id": f"diogomede/{slug}", "title": slug, "code_file": "notebook.ipynb",
            "language": "python", "kernel_type": "notebook", "is_private": True,
            "enable_gpu": True, "enable_tpu": False, "enable_internet": True,
            "machine_shape": "NvidiaTeslaT4", "dataset_sources": DATASETS,
            "competition_sources": [], "kernel_sources": [], "model_sources": []}


fonte = PASTA / "rotulador_9b" / "notebook.py"
escrever(fonte.read_text().replace("# <<COMUM_OLLAMA>>", COMUM), fonte.with_suffix(".ipynb"))

modelo_treino = (PASTA / "treino" / "notebook.py").read_text()
for nome, (base, quants) in TREINOS.items():
    pasta = PASTA / f"treino_{nome}"
    pasta.mkdir(exist_ok=True)
    texto = (modelo_treino.replace("<<MODELO_BASE>>", base).replace("<<NOME>>", nome)
             .replace("<<QUANTS>>", json.dumps(quants)))
    escrever(texto, pasta / "notebook.ipynb")
    slug = "cr-treino-" + nome.replace("_", "-")
    (pasta / "kernel-metadata.json").write_text(json.dumps(metadata(slug), indent=2))
