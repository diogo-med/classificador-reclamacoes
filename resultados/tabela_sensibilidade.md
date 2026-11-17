| Modelo | F1 macro: teste completo (n=500) | F1 macro: sem textos vistos no treino (n=462) | F1 macro: reponderado por nota (n=500) |
|---|---|---|---|
| TF-IDF + regressão logística | 0.683 | 0.676 | 0.669 |
| Qwen3.5-2B sem ajuste (Ollama q8_0) | 0.525 | 0.515 | 0.530 |
| Qwen3.5-9B sem ajuste (Ollama Q4_K_M) | 0.727 | 0.722 | 0.713 |
| Qwen3.5-9B (prompt de rotulagem, few-shot, lotes de 25) † | 0.819 | 0.815 | 0.804 |
| Qwen3.5-2B + LoRA (este projeto) | 0.803 | 0.799 | 0.795 |
| Qwen3.5-2B + LoRA, GGUF q8_0 no Ollama | 0.808 | 0.804 | 0.799 |

- **Teste completo**: o mesmo F1 macro da tabela de qualidade.
- **Sem textos vistos no treino**: exclui as 38 avaliações do teste cujo texto normalizado (minúsculas, sem acentos e pontuação) é idêntico ao de alguma avaliação do treino rotulado (as que entram no treino do LoRA e do TF-IDF). Textos curtos repetidos, como "muito bom", aparecem no treino e no teste e favorecem os modelos treinados.
- **Reponderado por nota**: o teste tem 62% de notas 1-3, contra 35% na população com texto (amostragem com peso maior nas notas baixas, D08); cada avaliação pesa (fração da nota na população) / (fração da nota no teste), ou seja, nota 1: 0.57, nota 2: 0.56, nota 3: 0.57, nota 4: 1.70, nota 5: 1.70. Aproxima o F1 macro na distribuição real de notas.
