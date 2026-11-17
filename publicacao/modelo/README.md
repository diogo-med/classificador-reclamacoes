---
base_model: Qwen/Qwen3.5-2B
datasets:
- Diogv20/classificador-reclamacoes-olist
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
model-index:
- name: classificador-reclamacoes-qwen3.5-2b
  results:
  - task:
      type: text-classification
      name: Classificação multirrótulo de reclamações
    dataset:
      type: Diogv20/classificador-reclamacoes-olist
      name: Olist reviews rotuladas (teste, rótulos provisórios (Gemini))
      split: test
    metrics:
    - type: f1
      name: F1 macro (categorias)
      value: 0.8034
    - type: f1
      name: F1 micro (categorias)
      value: 0.8560
    - type: accuracy
      name: Acurácia (sentimento)
      value: 0.9120
    - type: accuracy
      name: JSON válido e no esquema
      value: 1.0000
---

# classificador-reclamacoes-qwen3.5-2b

Qwen3.5-2B ajustado com LoRA (16 bits, Unsloth) para ler uma avaliação de cliente de e-commerce
em português e devolver JSON:

```json
{"sentimento": "negativo", "categorias": ["atraso", "atendimento"], "resumo": "..."}
```

Categorias: `atraso`, `nao_recebido`, `incompleto`, `produto_errado`, `defeito_qualidade`, `atendimento`, `elogio`, `outro`.

**É destilação:** os rótulos de treino foram gerados pelo Gemini 3.6 Flash; o modelo aprende a
imitá-lo. A pergunta do projeto é quanto da qualidade do Gemini um 2B retém, e a que custo.

Código, avaliação completa e decisões: [github.com/diogo-med/classificador-reclamacoes](https://github.com/diogo-med/classificador-reclamacoes).
Dataset: [Diogv20/classificador-reclamacoes-olist](https://huggingface.co/datasets/Diogv20/classificador-reclamacoes-olist).
Demo: [Diogv20/classificador-reclamacoes-demo](https://huggingface.co/spaces/Diogv20/classificador-reclamacoes-demo).

> ⚠️ **Rótulos de teste provisórios.** As métricas abaixo usam como referência os rótulos do próprio Gemini (o professor), ainda não revisados por humano. Como este modelo foi treinado para imitar o estilo de rotulagem do Gemini, sua vantagem sobre os outros modelos da tabela está **inflada** até a revisão; não leia a tabela como prova de que ele supera o Qwen3.5-9B.

## Resultados no teste

Teste: 500 avaliações; referência: rótulos **provisórios** do Gemini (revisão humana pendente).

| Modelo | F1 macro | F1 macro IC 95% | F1 micro | F1 `atraso` | Acerto exato | Sentimento (acc / F1 macro) | JSON válido (%) | JSON no esquema (%) | Atraso real: cobertura (%) |
|---|---|---|---|---|---|---|---|---|---|
| TF-IDF + regressão logística | 0.683 | [0.645, 0.719] | 0.750 | 0.692 | 0.646 | 0.874 / 0.766 | 100.0 | 100.0 | 33.3 |
| Qwen3.5-2B sem ajuste (Ollama q8_0) | 0.525 | [0.483, 0.563] | 0.617 | 0.382 | 0.442 | 0.896 / 0.745 | 100.0 | 98.0 | 72.2 |
| Qwen3.5-9B sem ajuste (Ollama Q4_K_M) | 0.727 | [0.689, 0.761] | 0.789 | 0.566 | 0.682 | 0.922 / 0.855 | 100.0 | 100.0 | 54.2 |
| Qwen3.5-9B (prompt de rotulagem, few-shot, lotes de 25) † | 0.819 | [0.784, 0.851] | 0.864 | 0.721 | 0.788 | 0.938 / 0.888 | 99.8 | 99.8 | 48.6 |
| Qwen3.5-2B + LoRA (este projeto) | 0.803 | [0.769, 0.838] | 0.856 | 0.811 | 0.782 | 0.912 / 0.803 | 100.0 | 100.0 | 38.9 |
| Qwen3.5-2B + LoRA, GGUF q8_0 no Ollama | 0.808 | [0.776, 0.841] | 0.859 | 0.819 | 0.786 | 0.912 / 0.797 | 100.0 | 100.0 | 37.5 |
| Gemini (prompt de rotulagem, lotes de 100) vs. humano | pendente de revisão | — | — | — | — | — | — | — | — | — |

Diferenças de F1 macro com IC de 95% (bootstrap pareado, 1000 reamostragens do teste com reposição, seed 42, percentis 2,5 e 97,5), medidas contra rótulos **provisórios do Gemini**, ou seja, diferença de concordância com o Gemini:

- Qwen3.5-2B + LoRA (este projeto) − Qwen3.5-9B sem ajuste (Ollama Q4_K_M): +0.076 [+0.042, +0.113]; primeiro maior em 100.0% das reamostragens
- Qwen3.5-2B + LoRA (este projeto) − TF-IDF + regressão logística: +0.120 [+0.087, +0.158]; primeiro maior em 100.0% das reamostragens
- Qwen3.5-2B + LoRA (este projeto) − Qwen3.5-9B (prompt de rotulagem, few-shot, lotes de 25): -0.016 [-0.046, +0.016]; primeiro maior em 16.6% das reamostragens

O intervalo cobre só a variação por amostragem das 500 avaliações do teste; não cobre erros dos rótulos de referência nem a variação entre treinos (um só seed).

† Os baselines "sem ajuste" recebem o prompt curto (`prompt_baseline`), que só lista os nomes das categorias, sem definições. A linha † é o mesmo Qwen3.5-9B com o prompt de rotulagem completo (definições + 10 exemplos), em lotes de 25, com decodificação restrita ao esquema JSON (por isso as colunas de JSON não são comparáveis com as outras linhas) e rodado na GPU do Kaggle (latência não medida na CPU local). Rotulou 499 das 500 avaliações do teste; a que falta conta como sem previsão. A diferença entre as duas linhas do 9B é efeito de prompt; a comparação 2B ajustado − 9B few-shot está na lista de diferenças.

**Enquanto a referência for o Gemini, a vantagem do 2B ajustado está inflada**: ele foi treinado para imitar o estilo de rotulagem do Gemini, e esta tabela mede concordância com o Gemini. O 9B e o TF-IDF não tiveram esse benefício na mesma medida (o 9B da linha † recebe as mesmas definições e exemplos que o Gemini, mas não aprendeu o estilo dele; o TF-IDF foi treinado com rótulos do Gemini, mas com muito menos capacidade). Esta tabela **não** mostra que o 2B ajustado supera o 9B; isso só pode ser afirmado depois de `src/recalcular.py` rodar com os rótulos revisados por humano.

A linha **Gemini (prompt de rotulagem, lotes de 100) vs. humano** fica pendente: antes da revisão, os rótulos de referência são os do próprio Gemini, e compará-los com eles mesmos daria F1 = 1,0 por definição. Ela é preenchida por `python src/recalcular.py` depois da revisão humana.

Rótulos do Gemini no teste: 28 em lotes de 25, 472 em lotes de 100 (ver DECISOES.md D19).

### Eficiência

Hardware local: 12th Gen Intel(R) Core(TM) i5-1235U, 12 CPUs lógicas, 15Gi de RAM, sem GPU dedicada (só CPU).

| Modelo | Onde | n | Latência média (s) | Latência p95 (s) | Tokens/s (geração) | JSON válido (%) | Custo por 1.000 avaliações |
|---|---|---|---|---|---|---|---|
| Qwen3.5-2B sem ajuste (q8_0) | Ollama local (CPU) | 100 | 7.21 | 10.02 | 10.9 | 100.0 | US$ 0 de API (hardware próprio) |
| Qwen3.5-9B sem ajuste (Q4_K_M) | Ollama local (CPU) | 100 | 26.14 | 36.37 | 3.4 | 100.0 | US$ 0 de API (hardware próprio) |
| Qwen3.5-2B + LoRA (q8_0) | Ollama local (CPU) | 100 | 5.96 | 7.27 | 11.2 | 100.0 | US$ 0 de API (hardware próprio) |
| Qwen3.5-2B + LoRA (q8_0, format=esquema) | Ollama local (CPU) | 100 | 6.07 | 7.62 | 10.8 | 100.0 | US$ 0 de API (hardware próprio) |
| Gemini 3.6 Flash (API) | API (rede) | 10 | 6.85 | 11.50 | — | 100.0 | US$ 2.995 |

- Mesma amostra fixa de 100 avaliações do teste para os modelos locais (`resultados/ids_eficiencia.txt`), uma chamada por avaliação, sem paralelismo, temperatura 0, raciocínio desligado, num_ctx 2048, 1 chamada de aquecimento descartada.
- Latência = tempo de parede da requisição HTTP ao Ollama local (inclui processar o prompt e gerar a resposta).
- Modelos locais: custo marginal de API = 0; o custo real é o hardware/energia, que não estimamos.
- Gemini 3.6 Flash (API): latência da API medida em 10 chamadas (nível gratuito, limitado a 20 requisições/dia), inclui rede; não é comparável diretamente à CPU local.
- Preço gemini-3.6-flash: US$ 0.75/1M tokens de entrada e US$ 3.75/1M de saída (https://aistudio.google.com/docs/pricing (redirecionado de ai.google.dev/gemini-api/docs/pricing), nível pago Standard, texto, consultado em 2026-11-08). Preço promocional válido até 31/12/2026; a partir de 01/01/2027 passa a US$ 1,50 (entrada) e US$ 7,50 (saída). A saída inclui tokens de raciocínio. Batch API: metade do preço. O nível gratuito não cobra, mas é limitado a 20 requisições/dia.

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
- Revisão humana pendente.
- Licença CC BY-NC-SA 4.0 herdada do dataset (uso não comercial). Os rótulos vêm da API do
  Gemini; confira os termos de uso da Google para saídas do modelo antes de qualquer uso.
