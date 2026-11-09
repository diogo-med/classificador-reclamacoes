Hardware local: 12th Gen Intel(R) Core(TM) i5-1235U, 12 CPUs lógicas, 15Gi de RAM, sem GPU dedicada (só CPU).

| Modelo | Onde | n | Latência média (s) | Latência p95 (s) | Tokens/s (geração) | JSON válido (%) | Custo por 1.000 avaliações |
|---|---|---|---|---|---|---|---|
| Qwen3.5-2B sem ajuste (q8_0) | Ollama local (CPU) | 100 | 7.21 | 10.02 | 10.9 | 100.0 | US$ 0 de API (hardware próprio) |
| Qwen3.5-9B sem ajuste (Q4_K_M) | Ollama local (CPU) | 100 | 26.14 | 36.37 | 3.4 | 100.0 | US$ 0 de API (hardware próprio) |
| Qwen3.5-2B + LoRA (q8_0) | Ollama local (CPU) | 100 | 5.96 | 7.27 | 11.2 | 100.0 | US$ 0 de API (hardware próprio) |
| Qwen3.5-2B + LoRA (q8_0, format=esquema) | Ollama local (CPU) | 100 | 6.07 | 7.62 | 10.8 | 100.0 | US$ 0 de API (hardware próprio) |

- Mesma amostra fixa de 100 avaliações do teste para os modelos locais (`resultados/ids_eficiencia.txt`), uma chamada por avaliação, sem paralelismo, temperatura 0, raciocínio desligado, num_ctx 2048, 1 chamada de aquecimento descartada.
- Latência = tempo de parede da requisição HTTP ao Ollama local (inclui processar o prompt e gerar a resposta).
- Modelos locais: custo marginal de API = 0; o custo real é o hardware/energia, que não estimamos.
- Preço gemini-3.6-flash: US$ 0.75/1M tokens de entrada e US$ 3.75/1M de saída (https://aistudio.google.com/docs/pricing (redirecionado de ai.google.dev/gemini-api/docs/pricing), nível pago Standard, texto, consultado em 2026-11-08). Preço promocional válido até 31/12/2026; a partir de 01/01/2027 passa a US$ 1,50 (entrada) e US$ 7,50 (saída). A saída inclui tokens de raciocínio. Batch API: metade do preço. O nível gratuito não cobra, mas é limitado a 20 requisições/dia.
