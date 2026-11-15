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
4. Comparação com TF-IDF + regressão logística (treinado com os mesmos rótulos do LoRA),
   Qwen3.5-2B e 9B sem ajuste (prompt curto) e o 9B com o prompt de rotulagem.

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
| Qwen3.5-2B sem ajuste (prompt curto) | 0,525 [0,483; 0,563] |
| Qwen3.5-9B sem ajuste (prompt curto) | 0,727 [0,689; 0,761] |
| Qwen3.5-9B com o prompt de rotulagem (few-shot, lotes de 25) † | 0,819 [0,784; 0,851] |
| Qwen3.5-2B + LoRA (prompt curto) | 0,803 [0,769; 0,838] |

† Mesmo 9B, com as definições e os 10 exemplos do prompt de rotulagem, em lotes de 25, com
decodificação restrita ao esquema, na GPU do Kaggle; 499 das 500 avaliações (a que falta conta
como sem previsão).

## O achado mais interessante: boa parte da diferença era o prompt
Os baselines "sem ajuste" recebem um prompt curto, que só lista os nomes das categorias. As
definições e as regras de fronteira (por exemplo, quando marcar `atraso`, `nao_recebido` ou os
dois) só existem no prompt com que o Gemini rotulou. O 2B ajustado também recebe o prompt curto,
mas aprendeu essas regras no treino; os baselines nunca as viram.

O 9B já tinha rotulado o teste com o prompt de rotulagem, como segundo rotulador. Com esse prompt,
ele passa de 0,727 para 0,819 de F1 macro em concordância com o Gemini. Contra o 2B ajustado:

- 2B ajustado − 9B com prompt curto: +0,076 [+0,042; +0,113];
- 2B ajustado − 9B com prompt de rotulagem: −0,016 [−0,046; +0,016], intervalo que inclui zero
  (o 2B ajustado fica à frente em só 16,6% das reamostragens).

(`resultados/tabela_qualidade.md`, `resultados/bootstrap_f1_macro.json`.) A comparação com o 9B
de prompt curto, que parecia uma vantagem clara do 2B ajustado, media em boa parte a diferença de
prompt. A leitura que se sustenta antes da revisão é outra: **o 2B ajustado, com um prompt de
~137 tokens, empata em concordância com o Gemini com um 9B de prompt longo**, e na mesma CPU leva
5,96 s por avaliação, contra 26,14 s do 9B com o prompt **curto** (~4,4×;
`resultados/tabela_eficiencia.md`). O 9B com o prompt longo não foi medido em CPU e seria mais
lento. Se o empate se confirmar com rótulos humanos, o que o ajuste compra é sobretudo
eficiência: as regras deixam de precisar estar no prompt, e um modelo menor dá conta.
Duas ressalvas: a linha † difere também no lote, na decodificação e no hardware; e o empate é
medido contra o Gemini, com o viés de estilo descrito acima, então pode mudar ou inverter com os
rótulos humanos. Falta um controle: o 2B **sem ajuste** com o prompt de rotulagem, que separaria
o efeito do prompt do efeito do LoRA no próprio 2B (não rodado).

## Sensibilidade
Duas suspeitas sobre o teste foram medidas (`resultados/tabela_sensibilidade.md`): 38 das 500
avaliações têm texto igual ao de alguma do treino ("muito bom"), e o teste tem 62% de notas 1-3,
contra 35% na população. Sem os textos repetidos, o F1 macro cai no máximo 0,010; reponderado para
a distribuição de notas da população, cai de 0,008 a 0,015 (sobe 0,005 no 2B sem ajuste). A ordem
dos modelos não muda em nenhum recorte.

## O que já se pode dizer sem a revisão
- **Formato:** o 2B ajustado respondeu JSON dentro do esquema em todas as 500 avaliações; o 2B
  sem ajuste, em 98% (`resultados/tabela_qualidade.md`). Isso não depende da referência.
- **Latência e custo** (`resultados/tabela_eficiencia.md`, mesma CPU, 100 avaliações fixas para os
  modelos locais): 2B ajustado 5,96 s em média (p95 7,27 s), 2B sem ajuste 7,21 s, 9B sem ajuste
  26,14 s, sem custo de API. O Gemini 3.6 Flash, medido em só 10 chamadas pela API (prompt curto,
  raciocínio no padrão do modelo), levou 6,85 s em média (p95 11,50 s) e custaria US$ 2,995 por
  1.000 avaliações ao preço atual. Quase todo o custo é raciocínio: média de 772 tokens de saída
  cobrados por chamada, numa delas 1.439 de raciocínio em 1.500. Com n=10, rede e nível gratuito,
  a comparação de latência é indicativa; desligar ou reduzir o raciocínio do Gemini mudaria
  latência e custo, e não foi testado.

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
- Baselines com prompt curto; o controle "2B sem ajuste com o prompt de rotulagem" não foi rodado.
- Latência do Gemini medida em só 10 chamadas.

## Próximo passo
Revisão humana do teste → `python src/recalcular.py` → reescrever as seções de resultado com a
tabela nova, incluindo a linha Gemini × humano, a diferença 2B ajustado − Gemini e a diferença
2B ajustado − 9B com o prompt de rotulagem. **As conclusões finais dependem dessa revisão.**
