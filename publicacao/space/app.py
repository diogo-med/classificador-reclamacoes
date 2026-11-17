"""Demo Gradio: cola uma avaliação, recebe o JSON. Roda o GGUF Q4_K_M em CPU com llama-cpp-python.

Local:  python app/app.py                      (usa CR_GGUF ou modelos/qwen2b_lora-Q4_K_M.gguf)
Space:  baixa o GGUF do repositório do modelo no Hugging Face (CR_HF_REPO).

Decodificação restrita: uma gramática gerada de esquema.ESQUEMA_SAIDA força o JSON do esquema
(chaves, sentimento e categorias da lista). CR_SEM_GRAMATICA=1 desliga (geração livre).
O prefixo do prompt e a temperatura 0 são os mesmos do treino (DECISOES.md D23).
"""
import json
import os
import sys
import time
import traceback
from pathlib import Path

import gradio as gr
from llama_cpp import Llama, LlamaGrammar

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # no Space, prompts.py fica ao lado
from esquema import ESQUEMA_SAIDA, avaliar_resposta, extrair_json  # noqa: E402
from prompts import prompt_chat  # noqa: E402

GGUF_LOCAL = Path(os.getenv("CR_GGUF", RAIZ / "modelos" / "qwen2b_lora-Q4_K_M.gguf"))
HF_REPO = os.getenv("CR_HF_REPO", "")
HF_ARQUIVO = os.getenv("CR_HF_ARQUIVO", "qwen2b_lora-Q4_K_M.gguf")
SEM_GRAMATICA = os.getenv("CR_SEM_GRAMATICA", "").strip().lower() in ("1", "true", "sim")
# n_ctx=1024: prompt (~150 tokens) + até 256 de saída; os textos do dataset têm no máximo ~230
# caracteres, então 1.500 caracteres (~500 tokens) cabem com folga.
MAX_CARACTERES = 1500

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


def carregar_gramatica():
    """Gramática GBNF do esquema de saída; None (geração livre) se desligada ou se falhar."""
    if SEM_GRAMATICA:
        print("Gramática desligada (CR_SEM_GRAMATICA): geração livre.", file=sys.stderr)
        return None
    try:
        return LlamaGrammar.from_json_schema(json.dumps(ESQUEMA_SAIDA), verbose=False)
    except Exception as e:  # noqa: BLE001
        print(f"Falha ao criar a gramática ({e!r}); seguindo com geração livre.", file=sys.stderr)
        return None


LLM = carregar_modelo()
GRAMATICA = carregar_gramatica()


def classificar(texto: str):
    texto = " ".join((texto or "").split())
    if not texto:
        return {}, "Cole uma avaliação."
    aviso = ""
    if len(texto) > MAX_CARACTERES:
        aviso = f" · entrada truncada: só os primeiros {MAX_CARACTERES} de {len(texto)} caracteres foram usados"
        texto = texto[:MAX_CARACTERES]
    inicio = time.time()
    try:
        # mesmo prefixo de chat do treino; decodificação gulosa (temperatura 0)
        out = LLM(prompt_chat(texto), max_tokens=256, temperature=0.0, stop=["<|im_end|>"],
                  grammar=GRAMATICA)
        bruto = out["choices"][0]["text"]
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        return {}, f"Erro ao classificar ({type(e).__name__}). Tente de novo com um texto mais curto.{aviso}"
    seg = time.time() - inicio
    av = avaliar_resposta(bruto)
    obj = extrair_json(bruto)
    status = (f"{seg:.1f} s · JSON {'válido' if av['json_valido'] else 'inválido'}"
              f"{' e dentro do esquema' if av['no_esquema'] else ''}{aviso}")
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
