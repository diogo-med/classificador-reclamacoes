# Progresso

Atualizado ao fim de cada etapa. Se a sessão cair, retomar a partir daqui.

## Estado final em 09/11/2026, 10:30: pronto para revisão

Todas as 16 etapas estão feitas ou documentadas. Falta só o que depende de você (lista no fim).
Critérios conferidos às 10:30:
- `python src/avaliar.py` regenera `resultados/` sem nenhuma diferença no git (reprodutível);
- `pytest`: 25 testes passando (após a coluna `conferido`, D36);
- app (`python app/app.py`, GGUF Q4_K_M) sobe e responde JSON no esquema (3,7 s e 2,8 s em dois
  exemplos), e foi encerrado;
- nada foi publicado: repositórios do Hugging Face não foram criados, notebooks e dataset do
  Kaggle continuam privados, repositório do GitHub continua privado.

> ⚠️ Todas as tabelas abaixo usam os **rótulos provisórios do Gemini** como referência. A vantagem
> do 2B ajustado está **inflada** (ele imita o estilo do Gemini; D34). Não leia como prova de que
> ele supera o 9B antes de `python src/recalcular.py` rodar com os seus rótulos.

### Qualidade no teste (`resultados/tabela_qualidade.md`)
Teste: 500 avaliações; referência: rótulos **provisórios** do Gemini (revisão humana pendente).

| Modelo | F1 macro | F1 macro IC 95% | F1 micro | F1 `atraso` | Acerto exato | Sentimento (acc / F1 macro) | JSON válido (%) | JSON no esquema (%) | Atraso real: cobertura (%) |
|---|---|---|---|---|---|---|---|---|---|
| TF-IDF + regressão logística | 0.683 | [0.645, 0.719] | 0.750 | 0.692 | 0.646 | 0.874 / 0.766 | 100.0 | 100.0 | 33.3 |
| Qwen3.5-2B sem ajuste (Ollama q8_0) | 0.525 | [0.483, 0.563] | 0.617 | 0.382 | 0.442 | 0.896 / 0.745 | 100.0 | 98.0 | 72.2 |
| Qwen3.5-9B sem ajuste (Ollama Q4_K_M) | 0.727 | [0.689, 0.761] | 0.789 | 0.566 | 0.682 | 0.922 / 0.855 | 100.0 | 100.0 | 54.2 |
| Qwen3.5-2B + LoRA (este projeto) | 0.803 | [0.769, 0.838] | 0.856 | 0.811 | 0.782 | 0.912 / 0.803 | 100.0 | 100.0 | 38.9 |
| Qwen3.5-2B + LoRA, GGUF q8_0 no Ollama | 0.808 | [0.776, 0.841] | 0.859 | 0.819 | 0.786 | 0.912 / 0.797 | 100.0 | 100.0 | 37.5 |
| Gemini (prompt de rotulagem, lotes de 100) vs. humano | pendente de revisão | — | — | — | — | — | — | — | — | — |

Diferenças de F1 macro com IC de 95% (bootstrap pareado, 1000 reamostragens do teste com reposição, seed 42, percentis 2,5 e 97,5), medidas contra rótulos **provisórios do Gemini**, ou seja, diferença de concordância com o Gemini:

- Qwen3.5-2B + LoRA (este projeto) − Qwen3.5-9B sem ajuste (Ollama Q4_K_M): +0.076 [+0.042, +0.113]; primeiro maior em 100.0% das reamostragens
- Qwen3.5-2B + LoRA (este projeto) − TF-IDF + regressão logística: +0.120 [+0.087, +0.158]; primeiro maior em 100.0% das reamostragens

O intervalo cobre só a variação por amostragem das 500 avaliações do teste; não cobre erros dos rótulos de referência nem a variação entre treinos (um só seed).

**Enquanto a referência for o Gemini, a vantagem do 2B ajustado está inflada**: ele foi treinado para imitar o estilo de rotulagem do Gemini, e esta tabela mede concordância com o Gemini. O 9B e o TF-IDF não tiveram esse benefício na mesma medida (o TF-IDF também foi treinado com rótulos do Gemini, mas com muito menos capacidade). Esta tabela **não** mostra que o 2B ajustado supera o 9B; isso só pode ser afirmado depois de `src/recalcular.py` rodar com os rótulos revisados por humano.

A linha **Gemini (prompt de rotulagem, lotes de 100) vs. humano** fica pendente: antes da revisão, os rótulos de referência são os do próprio Gemini, e compará-los com eles mesmos daria F1 = 1,0 por definição. Ela é preenchida por `python src/recalcular.py` depois da revisão humana.

Rótulos do Gemini no teste: 28 em lotes de 25, 472 em lotes de 100 (ver DECISOES.md D19).

### F1 por categoria (`resultados/tabela_f1_categorias.md`)
| Modelo | `atraso` | `nao_recebido` | `incompleto` | `produto_errado` | `defeito_qualidade` | `atendimento` | `elogio` | `outro` |
|---|---|---|---|---|---|---|---|---|
| TF-IDF + regressão logística | 0.69 | 0.77 | 0.72 | 0.71 | 0.65 | 0.66 | 0.93 | 0.33 |
| Qwen3.5-2B sem ajuste (Ollama q8_0) | 0.38 | 0.71 | 0.33 | 0.59 | 0.76 | 0.26 | 0.90 | 0.27 |
| Qwen3.5-9B sem ajuste (Ollama Q4_K_M) | 0.57 | 0.84 | 0.77 | 0.86 | 0.85 | 0.65 | 0.96 | 0.32 |
| Qwen3.5-2B + LoRA (este projeto) | 0.81 | 0.89 | 0.82 | 0.85 | 0.85 | 0.74 | 0.96 | 0.50 |
| Qwen3.5-2B + LoRA, GGUF q8_0 no Ollama | 0.82 | 0.90 | 0.82 | 0.85 | 0.85 | 0.75 | 0.96 | 0.52 |
| *suporte no teste* | 50 | 92 | 40 | 67 | 66 | 76 | 187 | 35 |

### Eficiência (`resultados/tabela_eficiencia.md`)
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

### Concordância Gemini × Qwen3.5-9B (`resultados/tabela_concordancia.md`)
Mesmo prompt de rotulagem; mede a ambiguidade de cada categoria, não quem está certo.

| Recorte | n | Sentimento (% / κ) | `atraso` (% / κ) | `nao_recebido` (% / κ) | `incompleto` (% / κ) | `produto_errado` (% / κ) | `defeito_qualidade` (% / κ) | `atendimento` (% / κ) | `elogio` (% / κ) | `outro` (% / κ) | Conjunto idêntico (%) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| teste | 499 | 94 / 0.90 | 92 / 0.68 | 97 / 0.89 | 99 / 0.91 | 98 / 0.90 | 96 / 0.82 | 92 / 0.70 | 99 / 0.97 | 94 / 0.50 | 79.0 |
| treino | 378 | 94 / 0.89 | 95 / 0.81 | 98 / 0.92 | 97 / 0.87 | 97 / 0.80 | 97 / 0.88 | 94 / 0.75 | 98 / 0.95 | 94 / 0.60 | 79.4 |

### Checagem de atraso pelas datas (`resultados/tabela_checagem_atraso.md`)
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

### O que está pronto
- Dados: 4.000 avaliações (treino 3.150 / validação 350 / teste 500); 2.050 rotuladas pelo Gemini
  (teste 500, validação 350, treino 1.200); 1.497 pelo Qwen3.5-9B.
- Checagem de lote (D29): lote 25 × 100 com 92% de conjunto idêntico contra 96% de piso de ruído.
- Qwen3.5-2B + LoRA treinado no Kaggle (17,7 min, T4), GGUF q8_0 e Q4_K_M em `modelos/` (fora do
  git), Ollama × Kaggle com rótulos idênticos em 20/20.
- Avaliação com TF-IDF, 2B e 9B sem ajuste, IC 95% por bootstrap pareado (D33), checagem de atraso
  nas duas direções (D35), análise de 10 erros (`resultados/analise_erros.md`).
- Planilha de revisão `data/revisao/revisao_manual.csv` (teste 500, treino aleatório 150,
  discordância 74) com regras de fronteira em `data/revisao/COMO_REVISAR.md`.
- `recalcular.py` testado de ponta a ponta numa cópia do repositório.
- README, `docs/artigo.md` (rascunho), ESTUDO.md, DECISOES.md (D01-D35), cards do Hugging Face em
  `publicacao/` (gerados localmente, sem envio).

### O que está pendente (e por quê)
- **Revisão humana** dos rótulos: só você pode fazer. Até lá, a linha "Gemini vs. humano" fica
  "pendente de revisão" e as comparações entre modelos ficam infladas a favor do 2B.
- **Latência do Gemini**: não medida; a cota de 9/11 (20 requisições) foi para a checagem de lote
  (2) e a rotulagem (18). A tabela de eficiência tem só os modelos locais e o preço oficial.
- **Baseline do Gemini com o prompt curto**: não rodado (D21; 500 chamadas = 25 dias de cota).
- **Treino restante (1.950 avaliações sem rótulo)**: fica sem rótulo; o 2B foi treinado com 1.200.
- **Seções de resultado do artigo**: reescrever depois do `recalcular.py`.

### Sua lista de tarefas
1. Revisar `data/revisao/revisao_manual.csv` seguindo `data/revisao/COMO_REVISAR.md` (bloco
   `teste` primeiro). Marque `x` na coluna `conferido` de **cada** linha que ler; o teste inteiro
   precisa de `x`. Correção vazia numa linha com `x` = aceita o sugerido. Salvar como CSV UTF-8
   separado por vírgula.
2. Rodar `python src/recalcular.py`. Ele valida a planilha, aplica as correções, mede Gemini ×
   humano, refaz tabelas, ICs e README e diz se é preciso retreinar.
3. Se ele recomendar retreino, rodar os comandos que ele imprime (Kaggle, cerca de 20 min de
   treino mais a fila) e depois `python src/avaliar.py`.
4. (Opcional, gasta 10 das 20 requisições do dia) `python src/eficiencia.py --gemini --n 10
   --nome "Gemini 3.6 Flash (API)"` e depois `python src/avaliar.py`.
5. Reler README, `docs/artigo.md` e `ESTUDO.md`; reescrever as conclusões do artigo com a tabela
   revisada (só afirmar que o 2B supera o 9B se o IC da diferença, agora contra rótulos humanos,
   excluir zero).
6. `python src/publicar.py --enviar` com `HF_TOKEN` no `.env` (cria os repositórios **privados**
   e envia GGUFs, adaptador, dataset e Space).
7. Revisar no Hugging Face e tornar públicos os repositórios que quiser; idem GitHub e, se
   quiser, os notebooks do Kaggle.

## Histórico

### 09/11/2026, 09:30

- 05:10: checagem de lote (D29): 25×100 92%, ruído 96%, queda 4 p.p. → lotes de 100.
- Rotulagem do Gemini concluída para a cota de hoje: **2.050 rótulos** (teste 500, validação 350,
  treino 1.200), 18 lotes de 100 sem nenhuma resposta inválida (2 erros 503 com nova tentativa).
- Dataset privado do Kaggle atualizado; kernel `diogomede/cr-treino-qwen2b-lora` v1 enviado
  (treino com 1.200, privado, T4). Monitor em segundo plano até terminar.
  Às 06:10 ainda na fila do Kaggle (esperando T4); às 07:08 rodando.
- **Treino do 2B concluído às 07:15** (17,7 min na T4, 1.200 exemplos, checkpoint-150). Validação
  (contra rótulos do Gemini): 100% no esquema, F1 macro 0,855, sentimento 92,9%. GGUFs q8_0 e
  Q4_K_M copiados para `modelos/` (fora do git).
- Ollama local × Kaggle (20 da validação): rótulos idênticos 20/20, texto 17/20.
- Previsões do teste com `cr-qwen2b-lora` no Ollama: 500/500 (08:25).
- `avaliar.py` rodado (provisório): F1 macro LoRA 0,803 (Kaggle) / 0,808 (Ollama q8_0), 9B 0,727,
  TF-IDF 0,683, 2B 0,525. Tabelas regeneradas no fim, depois da eficiência.
- Análise de 10 erros: `resultados/analise_erros.md`.
- Rodando: eficiência do `cr-qwen2b-lora` (livre e com format=esquema), 100 avaliações cada.
- Pedidos do usuário (09/11 manhã), feitos: IC 95% por bootstrap pareado (D33), aviso de vantagem
  inflada no README/tabela/artigo (D34), checagem de atraso nas duas direções com números absolutos
  e avaliações feitas antes da entrega (D35), regras de fronteira em `COMO_REVISAR.md`.
  `recalcular.py` testado numa cópia do repositório (sem tocar os arquivos reais). 18 testes passando.
- Rascunho do artigo: `docs/artigo.md` (números provisórios, com fonte).
- Planilha `data/revisao/revisao_manual.csv` gerada: teste 500, treino aleatório 150,
  discordância 74 (todas as disponíveis).

### 03:50

Conferido às 03:49: `cr-rotulador-9b` e `cr-treino-qwen08b-lora` com status COMPLETE e saídas
importadas; nenhum kernel rodando. Próximo passo: alarme das 05:10 → `checar_lote.py` → rotulagem.

### 08/11/2026, 23:40

### Bloqueio principal: cota do Gemini
O nível gratuito do `gemini-3.6-flash` permite **20 requisições/dia** (D19). Rotuladas até agora:
250 avaliações (202 treino, 28 teste, 20 validação). A cota renova por volta das **05:00 de 9/11**.
O restante vai em lotes de 100, com teste e validação primeiro.

### Feito
- Etapas 1-4: estrutura, dados (40.312 com texto), amostra 4.000 (3.150/350/500), categorias.
- Etapa 5 (parcial): 250/4.000 rotuladas.
- Etapa 6 (rodando): kernel `diogomede/cr-rotulador-9b` v2 (privado) no Kaggle: Qwen3.5-9B rotula
  teste + 1.000 do treino e roda os baselines 9B e 2B no teste. A v1 tinha `presence_penalty`
  1,5 (padrão do Modelfile) e foi substituída pela v2, com 0 (D22).
- Código pronto e testado (16 testes passando): `rotular.py`, `inferir.py`, `metricas.py`,
  `avaliar.py` (inclui TF-IDF), `revisao.py`, `recalcular.py`, `importar_kaggle.py`,
  `kaggle_dataset.py`, `eficiencia.py`, `publicar.py`, `app/app.py`, notebooks do Kaggle
  (`kaggle/treino/notebook.py` gera os kernels do 0.8B e do 2B).
- Dataset privado no Kaggle: `diogomede/classificador-reclamacoes-dados`.
- Preço oficial do Gemini 3.6 Flash salvo em `resultados/precos.json` (fonte e data).
- Eficiência local do `qwen3.5:2b` (100 avaliações fixas do teste): 7,2 s média, 10,0 s p95,
  10,9 tokens/s (`resultados/eficiencia.json`).
- Medição local do 9B: 28,6 s/avaliação (2ª tentativa; a 1ª morreu por memória).

### Concluído nesta noite
- Etapa 6: kernel `cr-rotulador-9b` v2 concluído. 9B rotulou 1.497 de 1.500 (3 inválidas);
  baselines 9B e 2B no teste (500 cada). Modelos idênticos aos locais (mesmos digests).
- Eficiência local: 2B 7,2 s (p95 10,0); 9B 26,1 s (p95 36,4). `resultados/eficiencia.json`.
- llama-cpp-python 0.3.36 carrega o Qwen3.5-2B público e gera JSON válido (D31).
- Linha do Gemini na tabela = "pendente de revisão" até o `recalcular.py` (D30).
- `src/checar_lote.py` pronto (D29): roda primeiro às 05:10.
- `avaliar.py` testado com os dados reais parciais (tabelas parciais não commitadas).

- Pipeline validado com o 0.8B (rótulos parciais, fora dos resultados, D28/D32): treino 5,4 min
  na T4, GGUF q8_0, prefixo idêntico ao da demo, Ollama × Kaggle com rótulos idênticos em 20/20.
  App Gradio testado com esse GGUF: subiu, respondeu (2,4 s, JSON no esquema) e foi encerrado.

### Rodando em segundo plano (pausa até a cota renovar)
- Alarme às 05:10 de 9/11: `checar_lote.py`, depois a rotulagem do Gemini.

### Falta
- Rotulagem do Gemini (9/11 de manhã), consolidar, nova versão do dataset do Kaggle.
- Treino 0.8B (pipeline) → treino 2B → GGUF → Ollama local → conferência em 20 exemplos.
- Planilha de revisão (depende dos rótulos do Gemini no teste), avaliação final, eficiência do 2B ajustado.
- Cards, README final, ESTUDO.md, artigo.

### Para retomar
```bash
conda activate classificador-reclamacoes
python -u src/checar_lote.py        # 1º: 2 requisições, só treino (D29); diz se usa lote 100 ou 50
python -u src/rotular.py --rotulador gemini --saida outputs/rotulos_gemini.jsonl --tamanho-lote 100 --pausa 15
kaggle kernels status diogomede/cr-rotulador-9b
python src/importar_kaggle.py rotulador_9b          # quando o kernel terminar
kaggle kernels push -p kaggle/treino_qwen08b_lora    # se ainda não tiver sido enviado
```
