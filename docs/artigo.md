# Rascunho: um 2B ajustado com LoRA no lugar de um LLM grande para classificar reclamações?

> **Rascunho com resultados provisórios.** Todos os números abaixo foram medidos contra rótulos
> do Gemini, antes da revisão humana. Eles serão substituídos pelos de `python src/recalcular.py`.
> Cada número cita o arquivo de `resultados/` de onde veio.

## A pergunta
Classificar avaliações de clientes em categorias de problema é uma tarefa típica para um LLM
grande via API. A pergunta do projeto: medido em rótulos revisados por um humano, quanto da
qualidade de um LLM grande (Gemini 3.6 Flash) um Qwen3.5-2B ajustado com LoRA retém, e a que custo
e latência, rodando numa CPU de notebook?

Dados: 4.000 avaliações reais da Olist (e-commerce brasileiro, 2016-2018; não é delivery de
comida), com peso maior nas notas 1-3. O teste (500) foi separado antes de qualquer leitura.

## O que foi feito
1. O Gemini rotulou 2.050 avaliações (teste, validação e 1.200 de treino) em lotes, com saída
   validada por esquema JSON. A cota gratuita (20 requisições/dia) limitou o treino a 1.200.
2. O Qwen3.5-9B rotulou as mesmas avaliações com o mesmo prompt, como segundo rotulador: a
   concordância entre os dois mede a ambiguidade de cada categoria
   (`resultados/concordancia_gemini_9b.json`).
3. LoRA 16 bits no Qwen3.5-2B com Unsloth numa T4 do Kaggle
   (`resultados/treino/qwen2b_lora/resultados_qwen2b_lora.json`), exportado em GGUF e servido
   pelo Ollama com o mesmo template do treino. Ollama × Kaggle em 20 exemplos:
   `resultados/conferencia_ollama_kaggle.json`.
4. Comparação com TF-IDF + regressão logística, Qwen3.5-2B e 9B sem ajuste.

## Antes de ler a tabela: por que a vantagem do 2B está inflada
Isto é **destilação**: o 2B aprendeu a imitar o Gemini. Enquanto a referência do teste também for
o Gemini, a tabela mede concordância com o professor, e o aluno é o único modelo treinado para
reproduzir o **estilo** de rotulagem dele (que categorias marcar juntas, onde passa a fronteira de
`atendimento` ou `outro`). O 9B e o 2B sem ajuste nunca viram esse estilo; o TF-IDF viu os mesmos
rótulos, mas com muito menos capacidade. Logo, a diferença a favor do 2B ajustado mistura
qualidade real com semelhança de estilo. **Este rascunho não afirma que o 2B ajustado supera o 9B
ou o TF-IDF.** Essa conclusão só pode sair da tabela refeita com os rótulos humanos.

## Resultados provisórios (concordância com o Gemini)
Fonte: `resultados/tabela_qualidade.md` e `resultados/bootstrap_f1_macro.json` (IC de 95% por
bootstrap pareado, 1.000 reamostragens do teste, seed 42).

| Modelo | F1 macro (IC 95%) contra o Gemini |
|---|---|
| TF-IDF + regressão logística | 0,683 [0,645; 0,719] |
| Qwen3.5-2B sem ajuste | 0,525 [0,483; 0,563] |
| Qwen3.5-9B sem ajuste | 0,727 [0,689; 0,761] |
| Qwen3.5-2B + LoRA | 0,803 [0,769; 0,838] |

A diferença 2B ajustado − 9B é de +0,076 [+0,042; +0,113] **em concordância com o Gemini**. O
intervalo exclui zero, mas ele só cobre a variação de amostragem do teste: não cobre o viés de
estilo descrito acima, que é justamente o que a revisão humana vai medir.

O que já se pode dizer sem a revisão:
- **Formato:** o 2B ajustado respondeu JSON dentro do esquema em todas as 500 avaliações; o 2B
  sem ajuste, em 98% (`resultados/tabela_qualidade.md`). Isso não depende da referência.
- **Custo e latência:** ver `resultados/tabela_eficiencia.md` (mesma CPU, 100 avaliações fixas).

## Uma checagem que não depende de LLM: atraso pelas datas
O dataset traz a data estimada e a data real de entrega, então dá para saber, sem LLM, quais
pedidos atrasaram e comparar com o rótulo `atraso` nas duas direções
(`resultados/tabela_checagem_atraso.md`). A surpresa: no teste, 62 dos 72 pedidos atrasados
foram avaliados **antes** de serem entregues (`resultados/avaliacao_antes_da_entrega.json`). O
cliente escreve "não recebi", não "atrasou". Por isso a cobertura de `atraso` sozinho é baixa
para todos os rotuladores, e a comparação justa é `atraso` **ou** `nao_recebido`: aí o 2B
ajustado marca 58 dos 72 atrasados, e 58 das 84 avaliações que ele marca tiveram atraso real.

## Onde o modelo erra
Em 10 erros sorteados (`resultados/analise_erros.md`), metade é erro claro do modelo e metade é
caso em que o rótulo do Gemini é discutível. As fronteiras instáveis são `atendimento` (pedido
de contato × queixa de atendimento) e `atraso` × `nao_recebido`.

## Limitações
- Domínio: e-commerce 2016-2018, não delivery de comida.
- Rótulos gerados por LLM; revisão humana pendente e, quando feita, com viés de ancoragem
  (o revisor vê o rótulo sugerido), o que também favorece o Gemini e o aluno dele.
- Um só seed e uma só divisão; categorias raras têm poucas dezenas de exemplos no teste.
- Treino limitado a 1.200 exemplos pela cota gratuita do Gemini.

## Próximo passo
Revisão humana do teste → `python src/recalcular.py` → reescrever as seções de resultado com a
tabela nova, incluindo a linha Gemini × humano e a diferença 2B ajustado − Gemini.
