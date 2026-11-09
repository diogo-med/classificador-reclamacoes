| Modelo | `atraso` | `nao_recebido` | `incompleto` | `produto_errado` | `defeito_qualidade` | `atendimento` | `elogio` | `outro` |
|---|---|---|---|---|---|---|---|---|
| TF-IDF + regressão logística | 0.69 | 0.77 | 0.72 | 0.71 | 0.65 | 0.66 | 0.93 | 0.33 |
| Qwen3.5-2B sem ajuste (Ollama q8_0) | 0.38 | 0.71 | 0.33 | 0.59 | 0.76 | 0.26 | 0.90 | 0.27 |
| Qwen3.5-9B sem ajuste (Ollama Q4_K_M) | 0.57 | 0.84 | 0.77 | 0.86 | 0.85 | 0.65 | 0.96 | 0.32 |
| Qwen3.5-2B + LoRA (este projeto) | 0.81 | 0.89 | 0.82 | 0.85 | 0.85 | 0.74 | 0.96 | 0.50 |
| Qwen3.5-2B + LoRA, GGUF q8_0 no Ollama | 0.82 | 0.90 | 0.82 | 0.85 | 0.85 | 0.75 | 0.96 | 0.52 |
| *suporte no teste* | 50 | 92 | 40 | 67 | 66 | 76 | 187 | 35 |
