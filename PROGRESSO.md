# Progresso

Atualizado ao fim de cada etapa. Se a sessão cair, retomar a partir daqui.

## Estado em 15/11/2026: análise conceitual (branch `analise-conceitual`, commits locais, sem push)

> ⚠️ A revisão humana dos rótulos do teste **ainda não foi feita**. Toda a referência é
> provisória (rótulos do Gemini); os números abaixo são concordância com o Gemini. Nada aqui
> afirma que um modelo supera outro.

Nada foi retreinado, re-inferido, enviado ao Kaggle nem publicado no Hugging Face. As mudanças
de código e de avaliação estão commitadas na branch `analise-conceitual` (sem push; a `main` não foi alterada).
Conferido:
- `python src/avaliar.py` continua idempotente (rodar duas vezes não muda nada) e os números
  antigos das tabelas não mudaram; só entraram linhas e tabelas novas;
- `pytest`: 47 testes passando (eram 25).

Tabelas atuais: `resultados/tabela_qualidade.md`, `resultados/tabela_f1_categorias.md`,
`resultados/tabela_sensibilidade.md`, `resultados/tabela_eficiencia.md`,
`resultados/tabela_concordancia.md`, `resultados/tabela_checagem_atraso.md` (e os blocos gerados
no README). Não estão copiadas aqui para não divergirem.

### O que foi feito hoje
- **Linha nova na tabela de qualidade:** "Qwen3.5-9B (prompt de rotulagem, few-shot, lotes de
  25) †", a partir dos rótulos que o 9B já tinha feito no teste como segundo rotulador (499 de
  500; `data/processed/rotulos_9b.parquet`), e o par 2B ajustado − 9B few-shot no bootstrap (D37).
- **Tabela de sensibilidade** (`resultados/tabela_sensibilidade.md`, bloco no README): teste
  completo, sem textos repetidos do treino e reponderado por nota (D38).
- **TF-IDF treina com os mesmos rótulos do LoRA**; o notebook de treino grava
  `arquivo_rotulos` (D39).
- **Regra de retreino por limiar em `recalcular.py`** (D40); `linhas_com_correcao` só conta
  mudança real de rótulo.
- **Bloco `status` gerado** no topo do README (provisório × revisado), refeito pelo
  `avaliar.py`.
- **Demo** (`app/app.py`): gramática gerada do esquema (ligada por padrão; `CR_SEM_GRAMATICA=1`
  desliga), limite de 1.500 caracteres e tratamento de erro (D41).
- **Cota do Gemini**: nova tentativa unificada em `rotular.chamar_com_retentativa`, parada limpa
  por cota diária também na inferência/eficiência, tokens de uso por lote na rotulagem (D43).
- **Latência do Gemini medida** (10 chamadas; D42), fechando a pendência de 09/11.
- Documentação: README (leitura dos resultados, como funciona, limitações), DECISOES.md D37-D43,
  ESTUDO.md (perguntas 1, 10, 12 a 15 e pontos fracos), `docs/artigo.md`,
  `data/revisao/COMO_REVISAR.md` ("Depois de revisar").

### Achados (contra os rótulos do Gemini)
- **Efeito de prompt.** O 9B com o prompt de rotulagem tem F1 macro 0,819 [0,784; 0,851] e acerto
  exato 0,788, contra 0,727 com o prompt curto e 0,803 do 2B ajustado. Diferença 2B ajustado − 9B
  few-shot: −0,016 [−0,046; +0,016], IC inclui zero (empate). A vantagem de +0,076 [+0,042;
  +0,113] sobre o "9B sem ajuste" era em boa parte efeito de prompt
  (`resultados/tabela_qualidade.md`, `resultados/bootstrap_f1_macro.json`).
- **Leitura honesta:** o 2B ajustado com o prompt curto (~137 tokens de entrada) empata, em
  concordância com o Gemini, com um 9B de prompt longo, e na mesma CPU é ~4,4× mais rápido que o
  9B com o prompt curto (5,96 s × 26,14 s); o 9B com o prompt longo não foi medido em CPU. A linha
  † rodou em lotes de 25, com decodificação restrita, na GPU do Kaggle; com rótulos humanos o
  empate pode mudar ou inverter.
- **Sensibilidade:** a ordem dos modelos não muda em nenhum recorte. Sem os 38 textos repetidos
  (n=462), o F1 macro cai no máximo 0,010; reponderado por nota (teste 62% de notas 1-3,
  população 35%), cai de 0,008 a 0,015 (sobe 0,005 no 2B sem ajuste)
  (`resultados/tabela_sensibilidade.md`).
- **Gemini pela API** (n=10, prompt curto, sem esquema, raciocínio no padrão): média 6,85 s, p95
  11,50 s, 100% JSON válido, média de 772 tokens de saída cobrados (quase tudo raciocínio: 1.439
  de 1.500 numa chamada), US$ 2,995 por 1.000 avaliações ao preço atual
  (`resultados/tabela_eficiencia.md`, `resultados/eficiencia/gemini_3_6_flash__api.jsonl`). O 2B
  ajustado em CPU (5,96 s) teve latência média menor nessa amostra pequena. Ressalvas: n=10, rede,
  nível gratuito; reduzir o raciocínio do Gemini mudaria latência e custo (não testado). Foram
  gastas 10 das 20 requisições do dia.
- **Demo com gramática** (teste local, GGUF Q4_K_M, 4 exemplos): mesmos rótulos e resumos com e
  sem gramática; 1,5 a 1,7× mais lenta (ex.: 3,2 s → 5,5 s). A gramática não impõe `uniqueItems`.
- **Cota:** o código antigo só detectava a cota diária por acaso (a regex capturava a mensagem
  inteira do erro).

### Observações (achados menores, não corrigidos)
- `src/publicar.py` marca `rotulo_provisorio=False` para **todas** as linhas quando existe
  `data/processed/rotulos_revisados.parquet`, inclusive linhas de treino que não foram revisadas.
  Corrigir antes de `publicar.py --enviar` se a revisão do treino for parcial.
- `kaggle/gerar_notebooks.py` gera ids de célula aleatórios a cada execução, o que produz ruído
  no diff dos `.ipynb`.
- As cópias de `src/` em `kaggle/dataset/` (fora do git) só são atualizadas ao rodar
  `src/kaggle_dataset.py`; rode-o antes de enviar um novo kernel.

### Limitações conhecidas e não resolvidas
- **Falta o 2B sem ajuste com o prompt de rotulagem completo**, que separaria o efeito do prompt
  do efeito do LoRA no próprio 2B. Experimento opcional (~1 a 1,5 h de CPU local); `src/inferir.py`
  hoje só usa o prompt curto, então exige uma opção nova no script. Você optou por não rodar agora.
- **Treino parcial:** só 1.200 das 3.150 avaliações de treino têm rótulo; rotular o resto custaria
  um dia de cota e exigiria retreino.
- **Um só seed de treino**; a variação entre treinos não é medida.
- **Revisão com ancoragem**: uma subamostra revisada às cegas mediria o viés; não está prevista.
- **Teste enriquecido em notas baixas**: agora há a coluna reponderada, mas a medida principal
  continua sendo a do teste como está.

### O que continua pendente
- **Revisão humana** dos rótulos do teste: só você pode fazer. Até lá, a linha "Gemini vs.
  humano" fica pendente e as comparações ficam infladas a favor do 2B ajustado.
- **Baseline do Gemini com o prompt curto no teste inteiro**: não rodado (D21; 500 chamadas).
- **Seções de resultado do artigo**: reescrever depois do `recalcular.py`.
- **Conferir e fazer merge da branch `analise-conceitual`.**

### Sua lista de tarefas
1. Revisar `data/revisao/revisao_manual.csv` seguindo `data/revisao/COMO_REVISAR.md` (bloco
   `teste` primeiro, inteiro, com `x` em `conferido`; se possível, pelo menos 50 linhas do
   `treino_aleatorio`, que é o que a regra de retreino usa). Salvar como CSV UTF-8 separado por
   vírgula.
2. Rodar `python src/recalcular.py`. Ele valida a planilha, aplica as correções, mede Gemini ×
   humano, refaz tabelas, ICs, o aviso de status e o README, e decide o retreino pela regra de
   limiar (D40): "retreino recomendado" só com pelo menos 50 linhas conferidas no
   `treino_aleatorio` e taxa de alteração de 10% ou mais; senão "retreino opcional" ou "amostra
   insuficiente". Os comandos são impressos em todos os casos.
3. Se ele recomendar retreino (ou se você quiser retreinar mesmo assim), rodar os comandos que
   ele imprime, na ordem (começam por `src/kaggle_dataset.py`, que atualiza as cópias em
   `kaggle/dataset/`, e terminam em `src/avaliar.py`; Kaggle, cerca de 20 min de treino mais a
   fila). O TF-IDF acompanha os rótulos do novo treino.
4. (Opcional) Experimento do 2B sem ajuste com o prompt de rotulagem completo: exige acrescentar
   uma opção de prompt a `src/inferir.py`; ~1 a 1,5 h de CPU local.
5. Reler README, `docs/artigo.md` e `ESTUDO.md`; reescrever as conclusões do artigo com a tabela
   revisada (só afirmar que o 2B supera o 9B, com qualquer dos dois prompts, se o IC da
   diferença, agora contra rótulos humanos, excluir zero).
6. Conferir o diff da branch `analise-conceitual` (`git diff main...analise-conceitual`) e fazer merge em `main`.
7. Antes de publicar, corrigir ou conferir a marcação `rotulo_provisorio` do `publicar.py`
   (observação acima). Depois, `python src/publicar.py --enviar` com `HF_TOKEN` no `.env` (cria
   os repositórios **privados** e envia GGUFs, adaptador, dataset e Space).
8. Revisar no Hugging Face e tornar públicos os repositórios que quiser; idem GitHub e, se
   quiser, os notebooks do Kaggle.

## Estado final em 09/11/2026, 10:30: pronto para revisão (superado pelo bloco de 15/11)

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

### Tabelas
As tabelas que ficavam copiadas aqui (qualidade, F1 por categoria, eficiência, concordância e
checagem de atraso) foram substituídas por remissão em 15/11, para não divergirem dos arquivos
gerados: ver `resultados/tabela_*.md` e o README. Os números de 09/11 não mudaram; em 15/11
entraram linhas e tabelas novas (ver o bloco de 15/11 acima).

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
