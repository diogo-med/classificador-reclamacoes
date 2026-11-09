"""Demo Gradio: cola uma avaliação, recebe o JSON. Roda o GGUF Q4_K_M em CPU com llama-cpp-python.

Local:  python app/app.py                      (usa CR_GGUF ou modelos/qwen2b_lora-Q4_K_M.gguf)
Space:  baixa o GGUF do repositório do modelo no Hugging Face (CR_HF_REPO).
"""
import json
import os
import sys
import time
from pathlib import Path

import gradio as gr
from llama_cpp import Llama

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # no Space, prompts.py fica ao lado
from esquema import avaliar_resposta, extrair_json  # noqa: E402
from prompts import prompt_chat  # noqa: E402

GGUF_LOCAL = Path(os.getenv("CR_GGUF", RAIZ / "modelos" / "qwen2b_lora-Q4_K_M.gguf"))
HF_REPO = os.getenv("CR_HF_REPO", "")
HF_ARQUIVO = os.getenv("CR_HF_ARQUIVO", "qwen2b_lora-Q4_K_M.gguf")

EXEMPLOS = [
    "Comprei 2 unidades e só chegou 1. Já mandei mensagem e ninguém responde.",
    "Produto ótimo, chegou antes do prazo. Recomendo!",
    "Até agora não recebi meu pedido e o prazo já passou faz uma semana.",
    "Veio com a tela quebrada e a cor diferente da que eu pedi.",
]


def carregar_modelo() -> Llama:
    caminho = GGUF_LOCAL
    if not caminho.exists() and HF_REPO:
        from huggingface_hub import hf_hub_download
        caminho = Path(hf_hub_download(HF_REPO, HF_ARQUIVO))
    if not caminho.exists():
        raise FileNotFoundError(f"GGUF não encontrado: {caminho}. Defina CR_GGUF ou CR_HF_REPO.")
    return Llama(model_path=str(caminho), n_ctx=1024, n_threads=os.cpu_count(), verbose=False)


LLM = carregar_modelo()


def classificar(texto: str):
    texto = " ".join((texto or "").split())
    if not texto:
        return {}, "Cole uma avaliação."
    inicio = time.time()
    # mesmo prefixo de chat do treino; decodificação gulosa (temperatura 0)
    out = LLM(prompt_chat(texto), max_tokens=256, temperature=0.0, stop=["<|im_end|>"])
    bruto = out["choices"][0]["text"]
    seg = time.time() - inicio
    av = avaliar_resposta(bruto)
    obj = extrair_json(bruto)
    status = (f"{seg:.1f} s · JSON {'válido' if av['json_valido'] else 'inválido'}"
              f"{' e dentro do esquema' if av['no_esquema'] else ''}")
    return (obj if isinstance(obj, dict) else {"resposta_crua": bruto}), status


with gr.Blocks(title="Classificador de reclamações (PT-BR)") as demo:
    gr.Markdown(
        "# Classificador de reclamações de e-commerce (PT-BR)\n"
        "Qwen3.5-2B ajustado com LoRA para devolver sentimento, categorias e resumo em JSON. "
        "Treinado por destilação de rótulos do Gemini em avaliações do Olist (2016-2018). "
        "Roda em CPU (GGUF Q4_K_M). Resultados e limitações no model card.")
    entrada = gr.Textbox(label="Avaliação do cliente", lines=4)
    botao = gr.Button("Classificar", variant="primary")
    saida = gr.JSON(label="Saída")
    status = gr.Markdown()
    gr.Examples(EXEMPLOS, inputs=entrada)
    botao.click(classificar, inputs=entrada, outputs=[saida, status])
    entrada.submit(classificar, inputs=entrada, outputs=[saida, status])

if __name__ == "__main__":
    demo.launch(server_name=os.getenv("GRADIO_SERVER_NAME", "127.0.0.1"))
