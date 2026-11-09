# Como revisar `revisao_manual.csv`

## O que é
Cada linha é uma avaliação com o rótulo sugerido pelo Gemini (`sentimento_sugerido`,
`categorias_sugeridas`). Sua tarefa é conferir e corrigir.

Ordem dos blocos (coluna `bloco`):
1. `teste` (500): **prioridade máxima**. É a referência de todas as métricas.
2. `treino_aleatorio` (150): sorteio do treino; mede a taxa de erro do Gemini no treino.
3. `treino_discordancia` (até 150; saíram 74, todas as discordâncias disponíveis no treino com
   rótulo dos dois, fora as já sorteadas): onde Gemini e Qwen3.5-9B discordaram; casos difíceis.
   Serve para corrigir rótulos de treino, **não** para estimar a taxa de erro (é enviesado).

Se faltar tempo, revise pelo menos o bloco `teste`.

## Regra de preenchimento
- **Coluna revisada vazia = aceito o sugerido.** Só preencha quando discordar.
- `sentimento_revisado`: `positivo`, `neutro` ou `negativo`.
- `categorias_revisadas`: a lista **completa** correta, separada por `;`
  (ex.: `atraso;nao_recebido`). Não escreva só a diferença.
- `observacao`: livre (dúvida, texto ambíguo etc.). Não altera rótulos.
- Não mude `review_id`, `bloco`, `nota`, `texto` nem as colunas sugeridas.
- Salve como CSV UTF-8, separado por vírgula (Google Sheets: Arquivo → Fazer download → CSV;
  LibreOffice: "Manter formato atual", UTF-8).

## Categorias

| Categoria | Definição | Exemplo (do treino) |
|---|---|---|
| `atraso` | O cliente reclama de demora ou de prazo descumprido: chegou depois do prazo, ou o prazo já passou e o pedido não chegou. Elogio ao prazo ou 'demorou, mas chegou no prazo' NÃO é atraso. | "Não cumpriu o prazo" |
| `nao_recebido` | O cliente diz que não recebeu o pedido (ainda não chegou, consta como entregue mas não chegou, extraviado). Se também disser que o prazo passou, marque atraso junto. | "Ainda nao recebi o produto" |
| `incompleto` | Chegou só parte do pedido: faltou item, unidade, peça ou acessório (inclusive manual). | "Somente foi entregue uma unidade, ao invés de duas." |
| `produto_errado` | Chegou produto diferente do comprado ou do anunciado: outro modelo, cor, tamanho, especificação, ou não original. | "Recebi o produto errado comprei um modelo e veio outro" |
| `defeito_qualidade` | Produto quebrado, com defeito, danificado, embalagem violada ou qualidade abaixo do esperado (frágil, mal acabado). | "Produto já veio com defeito não funciona" |
| `atendimento` | Problema com o atendimento da loja, vendedor ou transportadora: não responde, não resolve, dificuldade para trocar, devolver ou cancelar, cancelamento indevido, entregador mal-educado. | "Já mandei mensagens informando e até agora não tive respostas." |
| `elogio` | Avaliação só positiva ou satisfeita, sem nenhum problema citado. Nunca use junto com uma categoria de problema. | "Entrega mto rápida, antes do prazo. Produto excelente!" |
| `outro` | Problema que não se encaixa nas categorias acima (preço, nota fiscal, cobrança, estoque indisponível) ou texto sem conteúdo avaliável. | "LIVRO FICA 2X MAIS DO VALOR VENDIDO PELO AUTOR EM SUA PAGINA" |

Regras gerais:
- Marque **todas** as categorias que o texto cita; `elogio` só se não houver nenhum problema.
- Sentimento pelo tom do texto, não pela nota (a nota aparece só como contexto).
- O atraso real calculado pelas datas **não** aparece de propósito: ele é a checagem independente.

## Viés de ancoragem
Você revisa vendo o rótulo sugerido, o que tende a puxar para a concordância. Para reduzir:
leia o texto e decida mentalmente **antes** de olhar as colunas sugeridas. O viés continua
existindo e é declarado no README.

## Depois de revisar
```bash
python src/recalcular.py
```
Ele valida a planilha (aponta linhas com valor inválido), aplica as correções, mede a
concordância Gemini × humano, regenera métricas, tabelas e README e diz se é preciso retreinar.
