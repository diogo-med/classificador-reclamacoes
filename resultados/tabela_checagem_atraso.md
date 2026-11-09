| Fonte | Recorte | n (atraso conhecido) | Atrasados reais | `atraso`: atrasados marcados | `atraso`: marcados que atrasaram | `atraso` ou `nao_recebido`: atrasados marcados | `atraso` ou `nao_recebido`: marcados que atrasaram | κ (`atraso`) |
|---|---|---|---|---|---|---|---|---|
| Gemini (rótulos) | todas | 1881 | 294 (15.6%) | 125/294 (42.5%) | 125/182 (68.7%) | 232/294 (78.9%) | 232/348 (66.7%) | 0.46 |
| Gemini (rótulos) | teste | 456 | 72 (15.8%) | 24/72 (33.3%) | 24/38 (63.2%) | 57/72 (79.2%) | 57/86 (66.3%) | 0.37 |
| Qwen3.5-9B (rótulos) | todas | 1372 | 223 (16.3%) | 110/223 (49.3%) | 110/187 (58.8%) | 173/223 (77.6%) | 173/289 (59.9%) | 0.46 |
| Qwen3.5-9B (rótulos) | teste | 456 | 72 (15.8%) | 35/72 (48.6%) | 35/62 (56.5%) | 58/72 (80.6%) | 58/93 (62.4%) | 0.44 |
| TF-IDF + regressão logística | teste | 456 | 72 (15.8%) | 24/72 (33.3%) | 24/39 (61.5%) | 52/72 (72.2%) | 52/100 (52.0%) | 0.36 |
| Qwen3.5-2B sem ajuste (Ollama q8_0) | teste | 456 | 72 (15.8%) | 52/72 (72.2%) | 52/166 (31.3%) | 60/72 (83.3%) | 60/176 (34.1%) | 0.28 |
| Qwen3.5-9B sem ajuste (Ollama Q4_K_M) | teste | 456 | 72 (15.8%) | 39/72 (54.2%) | 39/87 (44.8%) | 61/72 (84.7%) | 61/121 (50.4%) | 0.38 |
| Qwen3.5-2B + LoRA (este projeto) | teste | 456 | 72 (15.8%) | 28/72 (38.9%) | 28/41 (68.3%) | 58/72 (80.6%) | 58/84 (69.0%) | 0.43 |
| Qwen3.5-2B + LoRA, GGUF q8_0 no Ollama | teste | 456 | 72 (15.8%) | 27/72 (37.5%) | 27/40 (67.5%) | 57/72 (79.2%) | 57/83 (68.7%) | 0.42 |

"Atrasados marcados" = dos pedidos que atrasaram pelas datas, quantos o rótulo marca (cobertura). "Marcados que atrasaram" = das avaliações com o rótulo, quantas tiveram atraso pelas datas (precisão).

No teste, 62 dos 72 pedidos atrasados pelas datas foram avaliados **antes** de serem entregues (`resultados/avaliacao_antes_da_entrega.json`, gerado por `src/datas_avaliacao.py`). Nesses casos o cliente escreve "não recebi", e o rótulo esperado é `nao_recebido` (com `atraso` só se ele disser que o prazo passou). Por isso a cobertura de `atraso` sozinho é baixa para todos os modelos, e a coluna `atraso` ou `nao_recebido` é a comparação mais justa.
