# Análise de erros do Qwen3.5-2B + LoRA (teste)

Gerado a partir de `resultados/analise_erros.json` (`python src/analise_erros.py`): 10 erros
sorteados com seed fixa entre as previsões do modelo ajustado (Kaggle, 16 bits) no teste.
**Referência: rótulos provisórios do Gemini**, sem revisão humana. "Erro" aqui é discordar do
Gemini, o que nem sempre é erro do modelo. Os comentários são leitura manual, não métrica.

| review | Gemini | Modelo | Leitura |
|---|---|---|---|
| 083cea85 | negativo, atraso | negativo, atraso + nao_recebido | Texto não diz se chegou. `nao_recebido` é plausível; ambíguo. |
| 2a67ce19 | negativo, produto_errado | negativo, defeito_qualidade | Cartucho falsificado. Fronteira entre as duas categorias; o Gemini parece mais certo (não é o produto anunciado). |
| 4a746391 | negativo, incompleto + nao_recebido | negativo, atraso + nao_recebido | Item de um pedido separado não chegou. Os dois concordam em `nao_recebido`; `incompleto` x `atraso` é ambíguo. |
| 603b745b | negativo, atendimento + incompleto | negativo, outro | "Paguei 3 e recebi 1": `incompleto` é claro. **Erro do modelo.** |
| 6724babb | negativo, atraso | negativo, atraso + nao_recebido | "Quase um mês para entregar" indica que chegou. **Erro do modelo** (`nao_recebido` a mais). |
| 6a4d0426 | neutro, defeito_qualidade | negativo, defeito_qualidade | Nota 2, elogio e crítica. Sentimento na fronteira neutro/negativo. |
| 75506434 | negativo, atendimento + nao_recebido | negativo, nao_recebido | "Nenhuma satisfação" sustenta `atendimento`. **Erro do modelo** (omissão). |
| 83d69188 | positivo, elogio | neutro, outro | Nota 5, recebeu certo e no prazo. **Erro do modelo**; o texto confuso ("não este o produto") o desviou. |
| cc588946 | negativo, atendimento + incompleto | negativo, incompleto | "Aguardo contato" é pedido de contato, não queixa de atendimento. O modelo parece mais certo. |
| e466966f | negativo, atraso | negativo, atendimento | Demora para postar = `atraso` pela definição. **Erro do modelo.** |

## Padrões
1. **`atendimento` é a fronteira mais instável**: o Gemini o acrescenta quando o cliente pede
   contato ou não teve resposta; o modelo às vezes omite (75506434) e às vezes acerta ao omitir
   (cc588946). Entre Gemini e 9B, no teste, está entre as de κ mais baixo (0,70), com `atraso` (0,68) e
   `outro` (0,50) (`resultados/tabela_concordancia.md`).
2. **`atraso` × `nao_recebido`**: o modelo tende a marcar os dois quando o texto fala de demora
   (083cea85, 6724babb). Por isso a precisão de `nao_recebido` cai e a cobertura do atraso real
   pelas datas fica parecida com a do Gemini (`resultados/tabela_checagem_atraso.md`).
3. **Erros claros do modelo** em 5 de 10 (603b745b, 6724babb, 75506434, 83d69188, e466966f);
   nos outros 5 o rótulo do Gemini é discutível ou ambos são defensáveis. Isso reforça que a
   métrica final depende da revisão humana.
4. Nenhum erro de formato: todas as respostas são JSON no esquema.

Nomes como "lannister" e "targaryen" são a anonimização das lojas feita pela própria Olist.
