# %% [markdown]
# # Treino LoRA (Unsloth) — classificador-reclamacoes
#
# Modelo base: `<<MODELO_BASE>>`. LoRA em 16 bits (sem QLoRA, que a documentação do Unsloth
# desaconselha para Qwen3.5). Entrada = prompt de baseline (`src/prompts.py`), saída = JSON
# do rótulo do Gemini (destilação). O chat template é aplicado com `enable_thinking=False`,
# igual na inferência. A validação decide a melhor época; o teste só é previsto no fim,
# sem nenhuma decisão baseada nele. Nenhuma chave entra aqui; tudo sai em /kaggle/working.

# %%
MODELO_BASE = "<<MODELO_BASE>>"
NOME = "<<NOME>>"
EPOCAS = 3
LR = 2e-4
R = 16
ALPHA = 16
MAX_SEQ = 512
SEED = 42

# %%
import glob, os, subprocess, sys, time, json
from pathlib import Path
subprocess.run("pip install -q --upgrade unsloth unsloth_zoo jsonschema", shell=True, check=True)
SRC = os.path.dirname(glob.glob("/kaggle/input/**/amostra.parquet", recursive=True)[0])
sys.path.insert(0, SRC)
os.environ["CR_DATA_PROCESSED"] = SRC
SAIDA = Path("/kaggle/working")
print(sorted(os.listdir(SRC)))
subprocess.run("nvidia-smi", shell=True)

# %%
from unsloth import FastLanguageModel  # importar antes de transformers/trl
import pandas as pd
import torch
from datasets import Dataset
from trl import SFTConfig, SFTTrainer

from esquema import avaliar_resposta
from metricas import metricas_categorias, metricas_sentimento, metricas_formato
from prompts import mensagens_baseline

amostra = pd.read_parquet(Path(SRC) / "amostra.parquet")
# rótulos revisados por humano (se já existirem) ou os do Gemini
arq_rotulos = Path(SRC) / "rotulos_revisados.parquet"
if not arq_rotulos.exists():
    arq_rotulos = Path(SRC) / "rotulos_gemini.parquet"
print("rótulos:", arq_rotulos.name)
rotulos = pd.read_parquet(arq_rotulos)
df = amostra.merge(rotulos[["review_id", "sentimento", "categorias", "resumo"]], on="review_id")
df["categorias"] = df["categorias"].apply(list)
print(df["split"].value_counts())

# %%
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODELO_BASE, max_seq_length=MAX_SEQ,
    load_in_4bit=False, load_in_16bit=True, full_finetuning=False)
# Qwen3.5 é multimodal: o objeto pode ser um processor; usamos o tokenizer de texto.
tok = getattr(tokenizer, "tokenizer", tokenizer)
model = FastLanguageModel.get_peft_model(
    model, r=R, lora_alpha=ALPHA, lora_dropout=0, bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    use_gradient_checkpointing="unsloth", random_state=SEED, max_seq_length=MAX_SEQ)

# %% [markdown]
# ## Formato: o mesmo prefixo no treino e na inferência
# `prefixo` = chat template com `add_generation_prompt=True` e `enable_thinking=False`
# (termina em `<|im_start|>assistant\n<think>\n\n</think>\n\n`). O alvo é o JSON + `<|im_end|>`.
# A perda só conta no alvo (formato prompt/completion do TRL).

# %%
def prefixo(texto):
    return tok.apply_chat_template(mensagens_baseline(texto), tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)

def alvo(linha):
    obj = {"sentimento": linha["sentimento"], "categorias": linha["categorias"], "resumo": linha["resumo"]}
    return json.dumps(obj, ensure_ascii=False) + "<|im_end|>"

exemplo = prefixo("Produto chegou quebrado.")
print(repr(exemplo))
(SAIDA / "prefixo_exemplo.txt").write_text(exemplo)

def para_dataset(parte):
    return Dataset.from_list([{"prompt": prefixo(t), "completion": alvo(l)}
                              for t, (_, l) in zip(parte["texto"], parte.iterrows())])

treino = para_dataset(df[df["split"] == "treino"])
valid = para_dataset(df[df["split"] == "validacao"])
print(len(treino), len(valid))

# %%
bf16 = torch.cuda.is_bf16_supported()
args = SFTConfig(
    output_dir=str(SAIDA / "checkpoints"), num_train_epochs=EPOCAS, learning_rate=LR,
    per_device_train_batch_size=8, gradient_accumulation_steps=2, per_device_eval_batch_size=8,
    lr_scheduler_type="linear", warmup_ratio=0.05, weight_decay=0.01,
    eval_strategy="epoch", save_strategy="epoch", save_total_limit=EPOCAS,
    load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False,
    logging_steps=10, seed=SEED, bf16=bf16, fp16=not bf16, max_length=MAX_SEQ,
    completion_only_loss=True, report_to="none", dataset_num_proc=1)
trainer = SFTTrainer(model=model, processing_class=tok, train_dataset=treino,
                     eval_dataset=valid, args=args)
inicio = time.time()
estatisticas = trainer.train()
tempo_treino = time.time() - inicio
historico = trainer.state.log_history
(SAIDA / "historico_treino.json").write_text(json.dumps(historico, indent=1))
print(f"treino: {tempo_treino / 60:.1f} min; melhor checkpoint: {trainer.state.best_model_checkpoint}")

# %% [markdown]
# ## Inferência (gulosa) na validação e no teste

# %%
FastLanguageModel.for_inference(model)
tok.padding_side = "left"

def gerar(textos, lote=16):
    saidas = []
    for i in range(0, len(textos), lote):
        prompts = [prefixo(t) for t in textos[i:i + lote]]
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=256, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        saidas += tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    return saidas

resultados = {"modelo_base": MODELO_BASE, "epocas": EPOCAS, "lr": LR, "r": R, "alpha": ALPHA,
              "tempo_treino_min": tempo_treino / 60, "melhor_checkpoint": trainer.state.best_model_checkpoint,
              "arquivo_rotulos": arq_rotulos.name,  # rotulos_gemini.parquet ou rotulos_revisados.parquet
              "n_treino": len(treino), "n_validacao": len(valid), "bf16": bf16,
              "gpu": torch.cuda.get_device_name(0)}
from importlib.metadata import version
resultados["versoes"] = {p: version(p) for p in ["unsloth", "unsloth_zoo", "transformers", "trl", "peft", "torch"]}
print(resultados["versoes"])
for split in ["validacao", "teste"]:
    parte = amostra[amostra["split"] == split].sort_values("review_id")
    inicio = time.time()
    respostas = gerar(parte["texto"].tolist())
    seg = time.time() - inicio
    with open(SAIDA / f"pred_{NOME}_{split}.jsonl", "w") as f:
        for rid, r in zip(parte["review_id"], respostas):
            f.write(json.dumps({"review_id": rid, "resposta": r, "modelo": NOME}, ensure_ascii=False) + "\n")
    resultados[f"tempo_inferencia_{split}_s"] = seg
    if split == "validacao":  # métricas só na validação; o teste é avaliado em src/avaliar.py
        av = [avaliar_resposta(r) for r in respostas]
        ouro = parte.merge(df[["review_id", "sentimento", "categorias"]], on="review_id")
        prev = dict(zip(parte["review_id"], av))
        p = [prev[i] for i in ouro["review_id"]]
        resultados["validacao"] = {
            **metricas_formato([x["json_valido"] for x in p], [x["no_esquema"] for x in p]),
            "categorias": metricas_categorias(ouro["categorias"], [x["categorias"] for x in p]),
            "sentimento": metricas_sentimento(ouro["sentimento"], [x["sentimento"] for x in p])}
        print(json.dumps({k: v for k, v in resultados["validacao"].items() if k != "categorias"}, indent=1),
              "F1 macro val:", resultados["validacao"]["categorias"]["f1_macro"])
(SAIDA / f"resultados_{NOME}.json").write_text(json.dumps(resultados, indent=1, ensure_ascii=False))

# %% [markdown]
# ## Salvar adaptador e exportar GGUF (q8_0 para comparação justa com o qwen3.5:2b do Ollama; Q4_K_M para a demo)

# %%
model.save_pretrained(str(SAIDA / f"adaptador_{NOME}"))
tok.save_pretrained(str(SAIDA / f"adaptador_{NOME}"))
for quant in <<QUANTS>>:
    try:
        inicio = time.time()
        model.save_pretrained_gguf(str(SAIDA / f"gguf_{NOME}"), tokenizer, quantization_method=quant)
        print(quant, "ok", (time.time() - inicio) / 60, "min")
    except Exception as e:
        print("falha no GGUF", quant, repr(e)[:1000])
subprocess.run(f"ls -laR {SAIDA} | grep -v checkpoint | head -60", shell=True)
# o Unsloth também deixa o modelo mesclado em 16 bits (safetensors, GB); só os GGUF interessam
subprocess.run(f"rm -rf {SAIDA}/checkpoints {SAIDA}/gguf_{NOME}/*.safetensors {SAIDA}/unsloth_compiled_cache", shell=True)
