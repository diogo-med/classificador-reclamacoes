# %% [markdown]
# # Segundo rotulador (Qwen3.5-9B) e baselines Qwen3.5 no teste
#
# Projeto classificador-reclamacoes. Este notebook:
# 1. instala o Ollama e baixa `qwen3.5:9b` e `qwen3.5:2b` (os mesmos modelos do Ollama local);
# 2. rotula teste + 1.000 do treino com o **mesmo prompt de rotulagem** do Gemini (segundo rotulador);
# 3. roda os baselines zero-shot (prompt de baseline, uma avaliação por chamada) no teste.
#
# Raciocínio desligado (`think: false`) e temperatura 0 em todas as chamadas.
# Não recebe nenhuma chave; tudo é salvo em /kaggle/working.

# %%
# <<COMUM_OLLAMA>>

# %%
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "jsonschema"], check=True)
import pandas as pd
from rotular import rotular
from inferir import inferir

amostra = pd.read_parquet(Path(SRC) / "amostra.parquet")
ids_9b = Path(SRC, "ids_9b.txt").read_text().split()
print(len(ids_9b), amostra.set_index("review_id").loc[ids_9b, "split"].value_counts().to_dict())
subir_ollama(["qwen3.5:9b", "qwen3.5:2b"])

# %% [markdown]
# ## 1. Segundo rotulador: Qwen3.5-9B com o prompt de rotulagem (lotes de 25)

# %%
inicio = time.time()
rotular(amostra[amostra["review_id"].isin(ids_9b)], "ollama", SAIDA / "rotulos_9b.jsonl",
        tamanho_lote=25, modelo="qwen3.5:9b")
print(f"rotulagem 9B: {(time.time() - inicio) / 60:.1f} min")

# %% [markdown]
# ## 2. Baselines zero-shot no teste (uma avaliação por chamada)

# %%
teste = amostra[amostra["split"] == "teste"].sort_values("review_id")
for modelo, nome in [("qwen3.5:9b", "qwen9b"), ("qwen3.5:2b", "qwen2b")]:
    inicio = time.time()
    inferir(teste, "ollama", modelo, SAIDA / f"pred_{nome}_teste.jsonl")
    print(f"baseline {modelo}: {(time.time() - inicio) / 60:.1f} min")

# %%
for arq in sorted(SAIDA.glob("*.jsonl")):
    print(arq.name, sum(1 for _ in open(arq)))
