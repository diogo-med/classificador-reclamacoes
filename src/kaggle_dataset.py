"""Monta e envia o dataset PRIVADO do Kaggle usado como fonte dos notebooks.

Conteúdo: amostra com splits, rótulos disponíveis, ids do segundo rotulador e
os módulos de src/ (os notebooks importam o mesmo código do repositório).
Nenhuma chave vai para o Kaggle.

Uso:
  python src/kaggle_dataset.py            # cria (1ª vez) ou publica nova versão
  python src/kaggle_dataset.py --so-montar
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATA_PROCESSED, RAIZ, SEED  # noqa: E402

PASTA = RAIZ / "kaggle" / "dataset"
SLUG = "diogomede/classificador-reclamacoes-dados"
MODULOS = ["config.py", "esquema.py", "prompts.py", "rotular.py", "inferir.py", "metricas.py"]
N_TREINO_9B = 1000


def ids_segundo_rotulador(amostra: pd.DataFrame) -> list[str]:
    """Teste inteiro + 1.000 do treino (seed fixa)."""
    teste = amostra[amostra["split"] == "teste"]["review_id"]
    treino = amostra[amostra["split"] == "treino"].sample(n=N_TREINO_9B, random_state=SEED)["review_id"]
    return sorted(teste) + sorted(treino)


def montar():
    if PASTA.exists():
        shutil.rmtree(PASTA)
    PASTA.mkdir(parents=True)
    amostra = pd.read_parquet(DATA_PROCESSED / "amostra.parquet")
    for arq in DATA_PROCESSED.glob("*.parquet"):
        shutil.copy(arq, PASTA / arq.name)
    (PASTA / "ids_9b.txt").write_text("\n".join(ids_segundo_rotulador(amostra)) + "\n")
    for m in MODULOS:
        origem = RAIZ / "src" / m
        if origem.exists():
            shutil.copy(origem, PASTA / m)
    meta = {
        "title": "classificador-reclamacoes-dados",
        "id": SLUG,
        "licenses": [{"name": "CC-BY-NC-SA-4.0"}],
        "subtitle": "Amostra rotulada de avaliações do Olist (privado, uso do projeto)",
        "description": ("Subconjunto de 4.000 avaliações do Brazilian E-Commerce Public Dataset "
                        "by Olist (CC BY-NC-SA 4.0), com divisão treino/validação/teste e rótulos "
                        "gerados por LLM (provisórios). Inclui o código de src/ do repositório "
                        "classificador-reclamacoes."),
    }
    (PASTA / "dataset-metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    print("montado em", PASTA, sorted(p.name for p in PASTA.iterdir()))


def enviar(mensagem: str):
    existe = subprocess.run(["kaggle", "datasets", "status", SLUG],
                            capture_output=True, text=True).returncode == 0
    if existe:
        cmd = ["kaggle", "datasets", "version", "-t", "-p", str(PASTA), "-m", mensagem]
    else:
        # sem -u/--public: o dataset é criado privado
        cmd = ["kaggle", "datasets", "create", "-t", "-p", str(PASTA)]
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--so-montar", action="store_true")
    p.add_argument("-m", "--mensagem", default="atualização")
    a = p.parse_args()
    montar()
    if not a.so_montar:
        enviar(a.mensagem)
