"""Baixa a saída de um kernel do Kaggle e copia o que importa para o repositório.

- rotulador_9b: rotulos_9b.jsonl -> data/processed/rotulos_9b.parquet;
                pred_qwen9b_teste.jsonl / pred_qwen2b_teste.jsonl -> resultados/predicoes/
- treino_<nome>: pred_<nome>_teste.jsonl e _validacao.jsonl -> resultados/predicoes/;
                 resultados_<nome>.json e historico_treino.json -> resultados/treino/<nome>/
                 (adaptador e GGUF ficam em outputs/kaggle/, fora do git)

Uso: python src/importar_kaggle.py rotulador_9b
     python src/importar_kaggle.py treino_qwen2b_lora
"""
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATA_PROCESSED, OUTPUTS, RESULTADOS  # noqa: E402
from rotular import consolidar  # noqa: E402

SLUGS = {"rotulador_9b": "diogomede/cr-rotulador-9b",
         "treino_qwen08b_lora": "diogomede/cr-treino-qwen08b-lora",
         "treino_qwen2b_lora": "diogomede/cr-treino-qwen2b-lora"}


def baixar(nome: str, pular_download=False) -> Path:
    destino = OUTPUTS / "kaggle" / nome
    destino.mkdir(parents=True, exist_ok=True)
    if not pular_download:
        subprocess.run(["kaggle", "kernels", "output", SLUGS[nome], "-p", str(destino), "-o"], check=True)
    return destino


def importar(nome: str, pasta: Path):
    pred = RESULTADOS / "predicoes"
    pred.mkdir(parents=True, exist_ok=True)
    if nome == "rotulador_9b":
        df = consolidar(pasta / "rotulos_9b.jsonl")
        df.to_parquet(DATA_PROCESSED / "rotulos_9b.parquet", index=False)
        print(f"{len(df)} rótulos do 9B")
        for m in ["qwen9b", "qwen2b"]:
            shutil.copy(pasta / f"pred_{m}_teste.jsonl", pred / f"{m}_teste.jsonl")
        shutil.copy(pasta / "ollama_modelos.txt", RESULTADOS / "ollama_modelos_kaggle.txt")
    else:
        modelo = nome.removeprefix("treino_")
        if modelo == "qwen08b_lora":
            # só validação do pipeline com rótulos parciais (D28): nada vai para resultados/
            print("0.8B: saídas mantidas só em", pasta)
            return
        for split in ["teste", "validacao"]:
            origem = pasta / f"pred_{modelo}_{split}.jsonl"
            if origem.exists():
                shutil.copy(origem, pred / f"{modelo}_{split}.jsonl")
        dest = RESULTADOS / "treino" / modelo
        dest.mkdir(parents=True, exist_ok=True)
        for arq in [f"resultados_{modelo}.json", "historico_treino.json", "prefixo_exemplo.txt"]:
            if (pasta / arq).exists():
                shutil.copy(pasta / arq, dest / arq)
    print("importado de", pasta)


if __name__ == "__main__":
    nome = sys.argv[1]
    importar(nome, baixar(nome, pular_download="--sem-download" in sys.argv))
