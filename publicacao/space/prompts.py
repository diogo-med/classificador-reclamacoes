"""Prompts versionados.

- PROMPT_ROTULAGEM: usado pelo Gemini (rotulador principal) e pelo
  Qwen3.5-9B (segundo rotulador). Em lote, com definições e exemplos.
  Todos os exemplos foram tirados do split de TREINO.
- PROMPT_BASELINE: zero-shot, curto, uma avaliação por chamada. Usado por
  todos os baselines LLM (Qwen3.5-2B, Qwen3.5-9B, Gemini) e também como
  entrada do modelo ajustado (no treino e na inferência), para que a
  comparação 2B sem ajuste x 2B ajustado use exatamente o mesmo prompt.

Ao mudar qualquer texto aqui, suba a versão correspondente.
"""
import json

VERSAO_ROTULAGEM = "rotulagem-v1"
VERSAO_BASELINE = "baseline-v1"

# Definição curta + exemplo (do treino) de cada categoria.
DEFINICOES = {
    "atraso": ("O cliente reclama de demora ou de prazo descumprido: chegou depois "
               "do prazo, ou o prazo já passou e o pedido não chegou. Elogio ao "
               "prazo ou 'demorou, mas chegou no prazo' NÃO é atraso.",
               "Não cumpriu o prazo"),
    "nao_recebido": ("O cliente diz que não recebeu o pedido (ainda não chegou, consta "
                     "como entregue mas não chegou, extraviado). Se também disser que o "
                     "prazo passou, marque atraso junto.",
                     "Ainda nao recebi o produto"),
    "incompleto": ("Chegou só parte do pedido: faltou item, unidade, peça ou acessório "
                   "(inclusive manual).",
                   "Somente foi entregue uma unidade, ao invés de duas."),
    "produto_errado": ("Chegou produto diferente do comprado ou do anunciado: outro "
                       "modelo, cor, tamanho, especificação, ou não original.",
                       "Recebi o produto errado comprei um modelo e veio outro"),
    "defeito_qualidade": ("Produto quebrado, com defeito, danificado, embalagem violada "
                          "ou qualidade abaixo do esperado (frágil, mal acabado).",
                          "Produto já veio com defeito não funciona"),
    "atendimento": ("Problema com o atendimento da loja, vendedor ou transportadora: "
                    "não responde, não resolve, dificuldade para trocar, devolver ou "
                    "cancelar, cancelamento indevido, entregador mal-educado.",
                    "Já mandei mensagens informando e até agora não tive respostas."),
    "elogio": ("Avaliação só positiva ou satisfeita, sem nenhum problema citado. "
               "Nunca use junto com uma categoria de problema.",
               "Entrega mto rápida, antes do prazo. Produto excelente!"),
    "outro": ("Problema que não se encaixa nas categorias acima (preço, nota fiscal, "
              "cobrança, estoque indisponível) ou texto sem conteúdo avaliável.",
              "LIVRO FICA 2X MAIS DO VALOR VENDIDO PELO AUTOR EM SUA PAGINA"),
}


def texto_definicoes() -> str:
    linhas = []
    for nome, (definicao, exemplo) in DEFINICOES.items():
        linhas.append(f'- {nome}: {definicao} Exemplo: "{exemplo}"')
    return "\n".join(linhas)


REGRAS_COMUNS = """Regras:
- sentimento: tom geral do texto (positivo, neutro ou negativo). Use neutro para texto misto equilibrado ou sem opinião. Baseie-se no texto, não em suposições.
- categorias: uma ou mais da lista, sem repetir. Marque todas as que o texto cita. "elogio" só quando não houver problema citado.
- resumo: uma frase curta em português (até 15 palavras) descrevendo o que o cliente disse.
- Nomes como lannister, stark, baratheon e targaryen são nomes anonimizados de lojas."""

# Exemplos do prompt de rotulagem: todos do split de treino (lidos na etapa 4).
EXEMPLOS_ROTULAGEM = [
    ("Meu pedido era para ser entregue até dia 22/03/18, até a presente data (24/03) a nota fiscal não foi emitida, solicitei várias vezes não obtive retorno, não recomendo esta Loja, nem a lannister!!!!!!",
     {"sentimento": "negativo", "categorias": ["atraso", "nao_recebido", "atendimento"],
      "resumo": "Prazo passou, pedido não chegou e a loja não responde."}),
    ("O produto demorou um pouco para chegar, mas chegou dentro do prazo estabelecido! Chegou de acordo com o contratado.",
     {"sentimento": "positivo", "categorias": ["elogio"],
      "resumo": "Demorou, mas chegou dentro do prazo e conforme contratado."}),
    ("Entrega de produto errado, comprei tapete para carro Ettios Rech modelo 2017, veio modelo Corolla",
     {"sentimento": "negativo", "categorias": ["produto_errado"],
      "resumo": "Recebeu tapete de modelo de carro diferente do comprado."}),
    ("Produto chegou com o assento rasgado, tentei entrar em contato com a lannister não obtive nenhuma resposta.",
     {"sentimento": "negativo", "categorias": ["defeito_qualidade", "atendimento"],
      "resumo": "Produto chegou rasgado e a loja não respondeu ao contato."}),
    ("Veio faltando mercadoria. Veio somente um relógio.",
     {"sentimento": "negativo", "categorias": ["incompleto"],
      "resumo": "Pedido chegou incompleto, veio só um relógio."}),
    ("A capinha é mais fina do que eu esperava. Mas no geral fica bonito no aparelho e ainda protege razoavelmente.",
     {"sentimento": "neutro", "categorias": ["defeito_qualidade"],
      "resumo": "Capinha mais fina que o esperado, mas bonita e protege."}),
    ("Não recebi. Não estou com o produto em mãos!",
     {"sentimento": "negativo", "categorias": ["nao_recebido"],
      "resumo": "Cliente diz que não recebeu o produto."}),
    ("muito bom obrigada",
     {"sentimento": "positivo", "categorias": ["elogio"],
      "resumo": "Cliente satisfeita."}),
    ("Comprei um produto na esperança de chegar na data ou antes para um presente e não chegou só tempo e dinheiro.",
     {"sentimento": "negativo", "categorias": ["atraso", "nao_recebido"],
      "resumo": "Produto não chegou a tempo para o presente."}),
    ("LIVRO FICA 2X MAIS DO VALOR VENDIDO PELO AUTOR EM SUA PAGINA",
     {"sentimento": "negativo", "categorias": ["outro"],
      "resumo": "Reclama que o preço é o dobro do vendido pelo autor."}),
]


def prompt_rotulagem(lote: list[tuple[str, str]]) -> str:
    """lote: lista de (id, texto). Retorna o prompt completo de rotulagem."""
    exemplos = "\n".join(
        f'Avaliação: "{t}"\nSaída: {json.dumps(s, ensure_ascii=False)}'
        for t, s in EXEMPLOS_ROTULAGEM)
    entrada = json.dumps([{"id": i, "texto": t} for i, t in lote], ensure_ascii=False, indent=0)
    return f"""Você é um analista de experiência do cliente. Classifique avaliações de clientes de um marketplace brasileiro de e-commerce.

Categorias (use só estes nomes):
{texto_definicoes()}

{REGRAS_COMUNS}

Exemplos:
{exemplos}

Classifique cada avaliação abaixo. Responda só com JSON no formato {{"avaliacoes": [{{"id": "...", "sentimento": "...", "categorias": [...], "resumo": "..."}}]}}, com um item para cada id, na mesma ordem.

Avaliações:
{entrada}"""


# ---------------------------------------------------------------------------
# Prompt de baseline (zero-shot, curto, uma avaliação). Também é a entrada do
# modelo ajustado.
INSTRUCAO_BASELINE = (
    "Classifique a avaliação de cliente de e-commerce abaixo. Responda só com JSON: "
    '{"sentimento": "positivo|neutro|negativo", "categorias": [...], "resumo": "..."}. '
    "Categorias possíveis (uma ou mais): atraso, nao_recebido, incompleto, "
    "produto_errado, defeito_qualidade, atendimento, elogio, outro. "
    "Use elogio só se não houver problema. Resumo: uma frase curta em português."
)


def prompt_baseline(texto: str) -> str:
    return f"{INSTRUCAO_BASELINE}\n\nAvaliação: {texto}"


def mensagens_baseline(texto: str) -> list[dict]:
    """Formato de chat usado no Ollama, no transformers e no treino."""
    return [{"role": "user", "content": prompt_baseline(texto)}]


# Prefixo de chat exatamente como o tokenizer do Qwen3.5 renderiza com
# add_generation_prompt=True e enable_thinking=False (conferido contra
# resultados/treino/<modelo>/prefixo_exemplo.txt, gerado no Kaggle).
# Usado pela demo (llama-cpp-python) e pelo Modelfile do Ollama.
PREFIXO_CHAT = "<|im_start|>user\n{conteudo}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def prompt_chat(texto: str) -> str:
    return PREFIXO_CHAT.format(conteudo=prompt_baseline(texto))
