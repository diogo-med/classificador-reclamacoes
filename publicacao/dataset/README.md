---
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

# classificador-reclamacoes-olist

Amostra de 4.000 avaliações com texto do
[Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)
(CC BY-NC-SA 4.0; atribuição: Olist). 2050 delas têm rótulo (sentimento, categorias de
problema e resumo); as demais têm as colunas de rótulo vazias (a cota gratuita do Gemini limitou a
rotulagem do treino):
  - train: 1200 de 3150
  - validation: 350 de 350
  - test: 500 de 500

## Origem e amostragem
- Deduplicação (uma avaliação por pedido, a mais recente), só avaliações com texto.
- Amostra estratificada por nota com peso 3 para notas 1-3 (onde estão as reclamações).
- Divisão fixa (seed 42): treino 3.150, validação 350, teste 500, feita **antes** de ler textos.
- `atraso_real`: calculado pelas datas (entrega real depois da estimada), independente dos rótulos.

## Como os rótulos foram gerados
- Gemini 3.6 Flash, temperatura 0, saída JSON validada contra esquema, prompt com definições e
  exemplos do treino (`src/prompts.py`, versão `rotulagem-v1`).
- Segundo rotulador (Qwen3.5-9B) em parte dos dados, para medir ambiguidade.
- Revisão humana **pendente**: rótulos do teste são provisórios.

## Limitações
- E-commerce de 2016-2018, não delivery de comida.
- Rótulos gerados por LLM; revisão humana pendente.
- Nomes de lojas aparecem anonimizados pela Olist (lannister, stark, ...).
