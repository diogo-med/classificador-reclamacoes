"""Etapa 13: prepara (e, só com --enviar, publica) modelo, dataset e Space no Hugging Face.

Sem --enviar: só gera os arquivos em publicacao/ para revisão (nada sai da máquina).
Com --enviar: precisa de HF_TOKEN no .env (ou no ambiente). Cria os repositórios
como PRIVADOS; torná-los públicos é um passo manual seu, depois de revisar.

Uso:
  python src/publicar.py                 # gera publicacao/{modelo,dataset,space}
  python src/publicar.py --enviar        # envia (privado)
"""
import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import CATEGORIAS, DATA_PROCESSED, RAIZ, RESULTADOS  # noqa: E402

PUB = RAIZ / "publicacao"
NOME_MODELO = "classificador-reclamacoes-qwen3.5-2b"
NOME_DATASET = "classificador-reclamacoes-olist"
NOME_SPACE = "classificador-reclamacoes-demo"
GITHUB = "https://github.com/diogo-med/classificador-reclamacoes"
MODELO_PRINCIPAL = "Qwen3.5-2B + LoRA (este projeto)"
GGUFS = {"q8_0": RAIZ / "modelos" / "qwen2b_lora-Q8_0.gguf",
         "q4_k_m": RAIZ / "modelos" / "qwen2b_lora-Q4_K_M.gguf"}
ADAPTADOR = RAIZ / "modelos" / "adaptador_qwen2b_lora"


def carregar_metricas() -> dict:
    caminho = RESULTADOS / "metricas.json"
    return json.loads(caminho.read_text()) if caminho.exists() else {}


def bloco_resultados_yaml(met: dict, usuario: str) -> str:
    """model-index com as métricas do modelo principal no teste."""
    r = met.get("modelos", {}).get(MODELO_PRINCIPAL)
    if not r:
        return ""
    ref = "revisados por humano" if met.get("referencia") == "revisada" else "provisórios (Gemini)"
    return f"""model-index:
- name: {NOME_MODELO}
  results:
  - task:
      type: text-classification
      name: Classificação multirrótulo de reclamações
    dataset:
      type: {usuario}/{NOME_DATASET}
      name: Olist reviews rotuladas (teste, rótulos {ref})
      split: test
    metrics:
    - type: f1
      name: F1 macro (categorias)
      value: {r['categorias']['f1_macro']:.4f}
    - type: f1
      name: F1 micro (categorias)
      value: {r['categorias']['f1_micro']:.4f}
    - type: accuracy
      name: Acurácia (sentimento)
      value: {r['sentimento']['acuracia']:.4f}
    - type: accuracy
      name: JSON válido e no esquema
      value: {r['json_no_esquema_pct'] / 100:.4f}
"""


def ler(arquivo: str) -> str:
    p = RESULTADOS / arquivo
    return p.read_text(encoding="utf-8") if p.exists() else "_ainda não gerado_\n"


def model_card(met: dict, usuario: str) -> str:
    ref_ok = met.get("referencia") == "revisada"
    aviso = ("" if ref_ok else
             "> ⚠️ **Rótulos de teste provisórios.** As métricas abaixo usam como referência os rótulos do "
             "próprio Gemini (o professor), ainda não revisados por humano. Como este modelo foi treinado "
             "para imitar o estilo de rotulagem do Gemini, sua vantagem sobre os outros modelos da tabela "
             "está **inflada** até a revisão; não leia a tabela como prova de que ele supera o Qwen3.5-9B.\n\n")
    return f"""---
base_model: Qwen/Qwen3.5-2B
datasets:
- {usuario}/{NOME_DATASET}
language:
- pt
license: cc-by-nc-sa-4.0
library_name: peft
pipeline_tag: text-generation
tags:
- lora
- unsloth
- gguf
- text-classification
- portuguese
- distillation
{bloco_resultados_yaml(met, usuario)}---

# {NOME_MODELO}

Qwen3.5-2B ajustado com LoRA (16 bits, Unsloth) para ler uma avaliação de cliente de e-commerce
em português e devolver JSON:

```json
{{"sentimento": "negativo", "categorias": ["atraso", "atendimento"], "resumo": "..."}}
```

Categorias: {", ".join(f"`{c}`" for c in CATEGORIAS)}.

**É destilação:** os rótulos de treino foram gerados pelo Gemini 3.6 Flash; o modelo aprende a
imitá-lo. A pergunta do projeto é quanto da qualidade do Gemini um 2B retém, e a que custo.

Código, avaliação completa e decisões: [{GITHUB.removeprefix("https://")}]({GITHUB}).
Dataset: [{usuario}/{NOME_DATASET}](https://huggingface.co/datasets/{usuario}/{NOME_DATASET}).
Demo: [{usuario}/{NOME_SPACE}](https://huggingface.co/spaces/{usuario}/{NOME_SPACE}).

{aviso}## Resultados no teste

{ler("tabela_qualidade.md")}
### Eficiência

{ler("tabela_eficiencia.md")}
## Arquivos
- `adaptador/`: adaptador LoRA (PEFT) sobre `Qwen/Qwen3.5-2B`.
- `qwen2b_lora-Q8_0.gguf`: mesma quantização do `qwen3.5:2b` do Ollama (comparação justa).
- `qwen2b_lora-Q4_K_M.gguf`: menor, usado na demo em CPU.
- `Modelfile`: para importar no Ollama com o mesmo chat template do treino.

## Como usar (Ollama)
```bash
ollama create cr-qwen2b-lora -f Modelfile
```
e chame `/api/chat` com `"think": false` e `temperature: 0`. O prompt de entrada é o
`prompt_baseline` do repositório (`src/prompts.py`).

## Limitações
- Dados de **e-commerce de 2016 a 2018** (Olist), não de delivery de comida.
- Rótulos gerados por LLM (Gemini); o modelo herda os vieses e erros do professor.
- Revisão humana {'feita vendo o rótulo sugerido (viés de ancoragem)' if ref_ok else 'pendente'}.
- Licença CC BY-NC-SA 4.0 herdada do dataset (uso não comercial). Os rótulos vêm da API do
  Gemini; confira os termos de uso da Google para saídas do modelo antes de qualquer uso.
"""


def dataset_card(met: dict, usuario: str, rotulados: dict | None = None) -> str:
    """rotulados: {split: (com rótulo, total)} contados nos parquets exportados."""
    rotulados = rotulados or {}
    linhas_rot = "\n".join(f"  - {s}: {c} de {n}" for s, (c, n) in rotulados.items())
    total_rot = sum(c for c, _ in rotulados.values())
    rev = RESULTADOS / "revisao_humana.json"
    concordancia = "Revisão humana **pendente**: rótulos do teste são provisórios."
    if rev.exists():
        r = json.loads(rev.read_text())["concordancia_gemini_humano"]
        partes = [f"{b}: conjunto de categorias idêntico em {v['conjunto_identico_pct']:.1f}% "
                  f"(n={v['n']}), sentimento {v['sentimento']['concordancia_pct']:.1f}%"
                  for b, v in r.items() if b != "treino_discordancia"]
        concordancia = "Concordância Gemini × humano — " + "; ".join(partes) + "."
    return f"""---
language:
- pt
license: cc-by-nc-sa-4.0
task_categories:
- text-classification
pretty_name: Avaliações Olist rotuladas (reclamações PT-BR)
size_categories:
- 1K<n<10K
source_datasets:
- olistbr/brazilian-ecommerce
---

# {NOME_DATASET}

Amostra de 4.000 avaliações com texto do
[Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)
(CC BY-NC-SA 4.0; atribuição: Olist). {total_rot} delas têm rótulo (sentimento, categorias de
problema e resumo); as demais têm as colunas de rótulo vazias (a cota gratuita do Gemini limitou a
rotulagem do treino):
{linhas_rot}

Código e avaliação: [{GITHUB.removeprefix("https://")}]({GITHUB}). Modelo treinado com estes
rótulos: [{usuario}/{NOME_MODELO}](https://huggingface.co/{usuario}/{NOME_MODELO}).

## Origem e amostragem
- Deduplicação (uma avaliação por pedido, a mais recente), só avaliações com texto.
- Amostra estratificada por nota com peso 3 para notas 1-3 (onde estão as reclamações).
- Divisão fixa (seed 42): treino 3.150, validação 350, teste 500, feita **antes** de ler textos.
- `atraso_real`: calculado pelas datas (entrega real depois da estimada), independente dos rótulos.

## Como os rótulos foram gerados
- Gemini 3.6 Flash, temperatura 0, saída JSON validada contra esquema, prompt com definições e
  exemplos do treino (`src/prompts.py`, versão `rotulagem-v1`).
- Segundo rotulador (Qwen3.5-9B) em parte dos dados, para medir ambiguidade.
- {concordancia}

## Limitações
- E-commerce de 2016-2018, não delivery de comida.
- Rótulos gerados por LLM{"; a revisão humana foi feita vendo o rótulo sugerido (viés de ancoragem)" if rev.exists() else "; revisão humana pendente"}.
- Nomes de lojas aparecem anonimizados pela Olist (lannister, stark, ...).
"""


def gerar(usuario: str):
    met = carregar_metricas()
    if PUB.exists():
        shutil.rmtree(PUB)
    (PUB / "modelo").mkdir(parents=True)
    (PUB / "dataset").mkdir()
    (PUB / "space").mkdir()

    (PUB / "modelo" / "README.md").write_text(model_card(met, usuario), encoding="utf-8")
    shutil.copy(RAIZ / "ollama" / "Modelfile", PUB / "modelo" / "Modelfile")

    amostra = pd.read_parquet(DATA_PROCESSED / "amostra.parquet")
    nome_rot = "rotulos_revisados" if (DATA_PROCESSED / "rotulos_revisados.parquet").exists() else "rotulos_gemini"
    rot = pd.read_parquet(DATA_PROCESSED / f"{nome_rot}.parquet")
    df = amostra.merge(rot, on="review_id", how="left")
    df["rotulo_provisorio"] = nome_rot == "rotulos_gemini"
    colunas = ["review_id", "nota", "texto", "sentimento", "categorias", "resumo", "atraso_real",
               "dias_atraso", "rotulo_provisorio"]
    for split, nome in [("treino", "train"), ("validacao", "validation"), ("teste", "test")]:
        df[df["split"] == split][colunas].to_parquet(PUB / "dataset" / f"{nome}.parquet", index=False)
    rotulados = {}
    for split, nome in [("treino", "train"), ("validacao", "validation"), ("teste", "test")]:
        d = df[df["split"] == split]
        rotulados[nome] = (int(d["sentimento"].notna().sum()), int(len(d)))
    (PUB / "dataset" / "README.md").write_text(dataset_card(met, usuario, rotulados), encoding="utf-8")

    for arq in ["app/app.py", "src/esquema.py", "src/prompts.py", "src/config.py"]:
        shutil.copy(RAIZ / arq, PUB / "space" / Path(arq).name)
    (PUB / "space" / "requirements.txt").write_text(
        "gradio==6.31.0\nhuggingface_hub==2.4.0\njsonschema==4.26.0\n"
        "--extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu\nllama-cpp-python==0.3.36\n")
    (PUB / "space" / "README.md").write_text(f"""---
title: Classificador de reclamações PT-BR
emoji: 📦
colorFrom: blue
colorTo: gray
sdk: gradio
sdk_version: 6.31.0
python_version: "3.11"
app_file: app.py
license: cc-by-nc-sa-4.0
models:
- {usuario}/{NOME_MODELO}
datasets:
- {usuario}/{NOME_DATASET}
---
Demo do modelo [{usuario}/{NOME_MODELO}](https://huggingface.co/{usuario}/{NOME_MODELO}) em CPU (GGUF Q4_K_M).
Rótulos de treino gerados pelo Gemini (revisão humana pendente). Código: [{GITHUB.removeprefix("https://")}]({GITHUB}).
""", encoding="utf-8")
    print("gerado em", PUB)


def enviar(usuario: str, token: str):
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    mid, did, sid = f"{usuario}/{NOME_MODELO}", f"{usuario}/{NOME_DATASET}", f"{usuario}/{NOME_SPACE}"
    api.create_repo(mid, repo_type="model", private=True, exist_ok=True)
    api.upload_folder(repo_id=mid, folder_path=str(PUB / "modelo"), repo_type="model")
    for quant, caminho in GGUFS.items():
        if caminho.exists():
            api.upload_file(path_or_fileobj=str(caminho), path_in_repo=caminho.name, repo_id=mid)
    if ADAPTADOR.exists():
        api.upload_folder(repo_id=mid, folder_path=str(ADAPTADOR), path_in_repo="adaptador")
    api.create_repo(did, repo_type="dataset", private=True, exist_ok=True)
    api.upload_folder(repo_id=did, folder_path=str(PUB / "dataset"), repo_type="dataset")
    api.create_repo(sid, repo_type="space", space_sdk="gradio", private=True, exist_ok=True)
    api.add_space_variable(sid, "CR_HF_REPO", mid)
    api.upload_folder(repo_id=sid, folder_path=str(PUB / "space"), repo_type="space")
    print(f"Enviado (privado): {mid}, {did}, {sid}. Revise e torne público no site do Hugging Face.")
    print("Obs.: um Space privado não consegue baixar um modelo privado sem token; "
          "torne o modelo público antes, ou adicione HF_TOKEN como secret do Space.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--enviar", action="store_true")
    p.add_argument("--usuario", default=None, help="usuário do HF (padrão: o dono do token)")
    a = p.parse_args()
    usuario = a.usuario or os.getenv("HF_USUARIO") or "SEU_USUARIO_HF"
    token = None
    if a.enviar:
        from dotenv import load_dotenv
        load_dotenv(RAIZ / ".env")
        token = os.getenv("HF_TOKEN")
        if not token:
            sys.exit("HF_TOKEN não encontrado no .env nem no ambiente.")
        if not a.usuario:
            from huggingface_hub import whoami
            usuario = whoami(token=token)["name"]
    gerar(usuario)
    if a.enviar:
        enviar(usuario, token)


if __name__ == "__main__":
    main()
