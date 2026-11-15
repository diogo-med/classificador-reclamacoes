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
