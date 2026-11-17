# Decisões

Decisões tomadas sem consulta, com alternativas e motivo. Onde o prompt de execução e o
`docs/plano.md` divergem, vale o prompt; as divergências estão marcadas com **[divergência]**.

## Gerais

**D01. Plano renomeado para `docs/plano.md`.** O arquivo chegou como
`docs/Classificador de reclamações em PT-BR (Olist) — base do projeto(1).md` (não versionado).
Renomeei para o nome que o prompt e o próprio plano usam.

**D02. [divergência] Pergunta central.** O plano pergunta se o modelo pequeno "iguala ou supera"
o LLM grande. O prompt redefine: *quanto da qualidade do Gemini um 2B ajustado retém, medido em
rótulos revisados por humano, e a que custo e latência*. Como os rótulos de treino vêm do Gemini,
isso é destilação: o aluno dificilmente supera o professor nos rótulos do próprio professor. Vale
a pergunta do prompt.

**D03. [divergência] Estrutura do repositório.** Além do plano: `tests/`, `ESTUDO.md`,
`src/config.py` (caminhos e seeds), `src/esquema.py` (esquema JSON e validação),
`src/amostrar.py` (etapa 3) e `data/processed/` (amostra rotulada versionada no git, pequena e
redistribuível sob CC BY-NC-SA 4.0). Alternativa: deixar tudo em `outputs/` (fora do git), mas aí
`src/avaliar.py` não rodaria do zero num clone.

**D04. Seed única = 42** em amostragem, divisão, embaralhamento de lotes e Ollama.

## Dados (etapa 2)

**D05. Deduplicação.** O plano diz "551 pedidos com mais de uma avaliação; manter a mais recente".
Nos dados baixados são 547 pedidos com mais de uma avaliação e 814 linhas com `review_id`
repetido (a mesma avaliação ligada a mais de um pedido). Regra: (1) por `order_id`, mantenho a de
`review_answer_timestamp` mais recente; (2) descarto `review_id` que continue ligado a mais de um
pedido, porque não dá para saber a qual pedido o atraso se refere, e isso contaminaria a checagem
independente. Alternativa: manter a primeira ocorrência. Removidas 1.690 linhas no total (1,7%).

**D06. Atraso real.** `atraso_real = 1` se a data (sem hora) de `order_delivered_customer_date` for
posterior à data de `order_estimated_delivery_date`; só para `order_status = delivered` com as duas
datas. Nos demais, fica nulo. Taxa nos entregues: 6,77% (o plano citava ~6,6%; a diferença vem da
comparação por data e da deduplicação).

**D07. Texto = título + ". " + mensagem.** Só entram avaliações com mensagem não vazia; o título
sozinho (geralmente 1-2 palavras) não basta. Não filtrei textos muito curtos ("ok", "recomendo"):
fazem parte da distribuição real.

## Amostragem (etapa 3)

**D08. Amostra de 4.000, cotas por nota proporcionais a contagem × peso**, com peso 3 para notas
1-3 e 1 para 4-5. Resultado: 1.494 / 365 / 613 / 343 / 1.185 (notas 1 a 5), ou seja 62% em 1-3
contra 35% na população com texto. Teste = 500 (estratificado por nota), validação = 10% do resto
(350), treino = 3.150. Feito por `src/amostrar.py` usando só a nota, antes de ler textos.

## Categorias (etapa 4)

Li 200 avaliações sorteadas só do treino (`random_state=7`). Mudanças em relação ao plano:

**D09. `sem_resposta` virou `atendimento`.** Nas 200, as queixas de atendimento iam além de "não
respondeu": dificuldade de trocar ou devolver, pedido de cancelamento ignorado, cancelamento
indevido, entregador mal-educado. Um nome mais amplo evita jogar tudo isso em `outro`.

**D10. Regra explícita entre `atraso` e `nao_recebido`.** "Ainda não recebi o produto" é a
reclamação mais comum e fica entre as duas. Regra: `nao_recebido` quando o cliente diz que não
recebeu; `atraso` quando reclama de demora ou prazo descumprido; as duas quando diz as duas coisas.
"Demorou, mas chegou no prazo" não é atraso. Como "ainda não recebi" em pedido entregue quase
sempre indica atraso real (a avaliação foi escrita antes da entrega), a checagem independente
reporta também `atraso OU nao_recebido`. Essa regra só está no prompt de rotulagem, não no prompt
curto dos baselines (ver D37).

**D11. `elogio` é exclusivo** (nunca junto com categoria de problema) e **`outro`** cobre problemas
fora da lista (preço, nota fiscal, cobrança, estoque) e textos sem conteúdo avaliável. Assim toda
avaliação tem pelo menos uma categoria.

**D12. Não criei categoria de cancelamento/cobrança.** Apareceu em ~4% das 200; com 3.150
exemplos de treino, uma classe tão rara teria F1 instável. Fica em `outro` ou `atendimento`.

**D13. Sentimento com 3 classes** (positivo, neutro, negativo), pelo tom do texto e não pela nota
(há casos como nota 1 com "GOSTEI MUITO...").

## Rotulagem (etapa 5)

**D14. Modelo do Gemini: `gemini-3.6-flash`** (versão retornada pela API em `model_version`:
`gemini-3.6-flash`). Testado em 08/11/2026: `gemini-2.5-flash` não está mais disponível para contas
novas (404); os aliases `gemini-3.5-flash` e `gemini-3.7-flash` redirecionavam para outras versões
(3.6 e 3.8); `gemini-3.8-flash` deu 503/504 repetidos. Fixei o nome exato `gemini-3.6-flash` para
reprodutibilidade. Configuração: temperatura 0, `response_mime_type=application/json`,
`response_json_schema` = esquema do lote, raciocínio no padrão do modelo.

**D15. Lotes de 25 avaliações**, embaralhadas com seed 42 (misturando splits, para o rotulador não
"saber" o que é teste). Validação item a item contra o esquema; itens inválidos ficam registrados
e podem ser repetidos depois.

**D16. `data/processed/amostra.parquet` entra no git.** São 524 KB com as 4.000 avaliações
sorteadas e a coluna `split`. Versionar fixa a divisão de teste: qualquer clone avalia nas mesmas
500 avaliações, mesmo se uma versão futura do pandas/sklearn mudar o sorteio com a mesma seed, e o
histórico do git prova que o teste foi separado antes da rotulagem e do treino. A licença do Olist
(CC BY-NC-SA 4.0) permite redistribuir com atribuição e os mesmos termos (ver `data/README.md`).
Alternativa: deixar fora e regenerar com `src/amostrar.py`, mas aí a divisão dependeria das versões
das bibliotecas. Os rótulos consolidados também vão para `data/processed/` pelo mesmo motivo; os
checkpoints crus da rotulagem (`outputs/*.jsonl`) ficam fora do git.

**D17. Checkpoints da rotulagem ficam em `outputs/`** (fora do git), um `.jsonl` por rotulador,
uma linha por lote. Isso não é `data/interim/`; por isso a conferência após a queda da sessão não
os encontrou lá.

**D18. Qwen3.5-9B não roda localmente.** A medição de 20 avaliações do 9B no Ollama local foi
encerrada pelo sistema (código 137, falta de memória) com o VS Code aberto e os 16 GB de RAM
compartilhados. O 9B como segundo rotulador e como baseline de qualidade roda no Kaggle (como o
prompt já pede para o rotulador). A tabela de eficiência local do 9B será tentada de novo com a
máquina mais livre; se falhar outra vez, fica documentada como bloqueada.

## Medições antes de lotes locais (regra das 2 horas)

| Modelo | Amostra | Latência média | Extrapolação | Decisão |
| --- | --- | --- | --- | --- |
| qwen3.5:2b (q8_0, Ollama local, CPU) | 20 avaliações da validação | 9,6 s (1ª chamada 25 s, com carga do modelo) | 500 × 9,6 s ≈ 80 min | abaixo de 2 h: baseline 2B pode rodar local |
| qwen3.5:9b (Q4_K_M, Ollama local, CPU), 1ª tentativa | 20 avaliações da validação | não medida (morto por memória, código 137) | — | — |
| qwen3.5:9b (Q4_K_M, Ollama local, CPU), 2ª tentativa | 20 avaliações da validação | 28,6 s (27,0 s sem a 1ª) | teste completo: 500 × 28,6 s ≈ 4 h; eficiência: 100 × 28,6 s ≈ 48 min | baseline no teste e rotulagem → Kaggle; eficiência (100) → local |
| cr-qwen2b-lora (LoRA, q8_0, Ollama local, CPU) | 20 avaliações da validação (`conferir_ollama.py`) | ≈ 6,4 s (2 min 8 s para as 20, incluindo `ollama create` e a carga) | 500 × 6,4 s ≈ 53 min | abaixo de 2 h: previsões do teste via Ollama rodam local |

Medições feitas na validação (não no teste) para não olhar saídas do teste.

## Cota do Gemini

**D19. [divergência] Lotes de 100 em vez de 20-30.** O nível gratuito do `gemini-3.6-flash`
permite **20 requisições por dia** por modelo (`quotaId:
GenerateRequestsPerDayPerProjectPerModel-FreeTier`, `quotaValue: 20`, conferido no erro 429 em
08/11/2026; renova por volta das 05:00, horário de Brasília). Em 08/11 foram feitas 10 requisições
de 25 = 250 avaliações (202 treino, 28 teste, 20 validação); o resto da cota foi gasto em testes de
conexão. Com lotes de 25, as 3.750 restantes levariam 7,5 dias. Decisão: o restante vai em lotes
de 100 (20 requisições = 2.000 avaliações por dia), na ordem teste → validação → treino.
Alternativas descartadas: (a) trocar o professor por um modelo `flash-lite` com cota maior, que é
mais fraco e misturaria professores; (b) parar em 750 avaliações por dia, abaixo do mínimo de 2.000
do prompt. Risco: lote maior pode piorar a atenção do modelo por item. Mitigação: todos os itens
são validados contra o esquema e a concordância com o 9B e com a revisão humana é medida igual
para todos; o tamanho do lote fica registrado por linha no checkpoint.

**D20. Orçamento de requisições do Gemini até 10/11.** 9/11: 20 requisições para rotulagem (teste
e validação completos, mais ~1.200 do treino). 10/11: o baseline do Gemini com uma avaliação por
chamada no teste (500 requisições) é **inviável** com 20 por dia. Ver D21 quando for executado.

**D21. Linha do Gemini na tabela de qualidade.** O baseline do Gemini com o prompt de baseline
(uma avaliação por chamada) precisaria de 500 requisições, ou 25 dias de cota gratuita. Fica
**bloqueado** e com o comando pronto em PROGRESSO.md. No lugar dele, a tabela traz os próprios
rótulos do Gemini (prompt de rotulagem: few-shot, em lote) como linha de referência do professor.
Enquanto o teste não for revisado, essa linha é 100% por construção (marcada como favorecida);
depois da revisão, ela mede exatamente o que a pergunta central precisa: a qualidade do professor
nos rótulos humanos. A latência do Gemini usa poucas chamadas avulsas (cota), com n declarado
(medida em 15/11, ver D42).

**D22. `presence_penalty = 0` em todas as chamadas ao Ollama.** O Modelfile oficial do `qwen3.5`
traz `presence_penalty 1.5`, `temperature 1`, `top_k 20` e `top_p 0.95`. Com temperatura 0, os
três últimos não importam, mas a penalidade de presença continua alterando os logits e penaliza
os tokens repetidos que todo JSON tem (aspas, chaves). O modelo ajustado importado com um
Modelfile próprio não teria essa penalidade, o que tornaria a comparação injusta. Zerei para
todos. A primeira versão do kernel do 9B rodou com o padrão e foi descartada (só a v2 é usada).

**D23. Formato do treino: prompt/completion com o prefixo do chat template.** O texto de treino é
`apply_chat_template([user], add_generation_prompt=True, enable_thinking=False)` (que termina em
`<|im_start|>assistant\n<think>\n\n</think>\n\n`) + JSON + `<|im_end|>`, e a perda só conta no
JSON. Assim o prefixo do treino é idêntico ao da inferência (transformers, Ollama e llama.cpp).
Alternativa: aplicar o template na conversa completa (user + assistant). Descartei porque o
template do Qwen3.5 trata o bloco `<think>` de forma diferente em turnos antigos e no turno de
geração, e o risco de divergência é exatamente o que o Unsloth aponta como causa mais comum de
modelo exportado ruim.

**D24. Entrada do modelo ajustado = prompt de baseline.** O 2B ajustado recebe o mesmo prompt
curto que o 2B sem ajuste. A diferença entre as duas linhas da tabela mede só o efeito do LoRA.

**D25. Licença do modelo e do dataset: CC BY-NC-SA 4.0.** O Qwen3.5 é Apache-2.0, mas o modelo é
treinado em dados CC BY-NC-SA 4.0 (Olist), e o dataset publicado é derivado deles. Uso a licença
mais restritiva nos dois. Os rótulos vêm da API do Gemini; os termos de uso da Google sobre
saídas do modelo ficam citados como ponto de atenção nos cards.

**D26. TF-IDF: C escolhido na validação** entre {0,3; 1; 3; 10} pelo F1 macro, com
`class_weight="balanced"`, n-gramas 1-2 e pelo menos uma categoria por avaliação (a de maior
probabilidade, quando nenhuma passa de 0,5), igual aos LLMs.

**D27. Notebooks do Kaggle como `.py` (jupytext) convertidos em `.ipynb`** por
`kaggle/gerar_notebooks.py`. Diferenças de código ficam legíveis no git, e um único modelo de
notebook gera os kernels do 0.8B e do 2B. O código de `src/` vai junto no dataset privado, para
o notebook usar exatamente o mesmo prompt, esquema e métricas do repositório.

**D28. Treino do Qwen3.5-0.8B só valida o pipeline.** Roda com os 250 rótulos parciais
disponíveis em 08/11 (202 de treino, 20 de validação), só para testar de ponta a ponta instalação,
formato, treino, inferência e exportação GGUF antes de gastar GPU no 2B. **Esses rótulos são
parciais e esse treino não entra em nenhum resultado:** a linha do 0.8B foi retirada de
`src/avaliar.py`, e suas saídas ficam só em `outputs/`.

**D29. Checagem de consistência dos lotes de 100, só com treino.** Até 08/11 à noite **nenhuma**
requisição em lote de 100 tinha sido feita: as 250 avaliações rotuladas foram todas em lotes de 25,
e a decisão D19 não tinha sido testada em nenhum split (muito menos no teste). Antes de rotular o
restante, `src/checar_lote.py` gasta **2 requisições** (de no máximo 3 permitidas) com avaliações
**só de treino** que já têm rótulo em lote de 25:
- A: 100 delas num único lote de 100 → concordância lote 25 × lote 100;
- B: 25 delas de novo num lote de 25 → piso de ruído (o Gemini não é perfeitamente determinístico
  mesmo com temperatura 0).

Regra fixada **antes** de ver o resultado: se o "conjunto de categorias idêntico" de 25 × 100 ficar
mais de 10 p.p. abaixo do piso de ruído, o restante usa lotes de 50. Uso 2 requisições em vez de 3
para preservar o mínimo de 2.000 rotuladas em 9/11 (250 + 18 × 100 = 2.050). Os rótulos dessa
checagem ficam em `outputs/checagem_lote/` e não substituem os originais. Resultado em
`resultados/checagem_lote.json`.

**Resultado (09/11, 05:10):** lote 25 × lote 100 (n=100): conjunto de categorias idêntico
92,0%, sentimento 97,0%. Piso de ruído 25 × 25 (n=25): 96,0% e 92,0%. Queda de 4,0 p.p., abaixo
do limiar de 10 → o restante segue em **lotes de 100**. Ressalva: o piso tem só 25 avaliações,
então cada avaliação vale 4 p.p.; a checagem descarta uma degradação grande, não uma pequena.

**D30. Linha do Gemini na tabela de qualidade = "pendente de revisão" até o `recalcular.py`.**
Com a referência provisória (rótulos do Gemini), comparar os rótulos do Gemini com eles mesmos dá
F1 = 1,0 por definição, um número sem significado que não deve aparecer. A linha
"Gemini (prompt de rotulagem, lotes de 100) vs. humano" só é calculada quando existem
`data/processed/rotulos_revisados.parquet` (gerado pelo `recalcular.py`). A tabela também informa
quantas avaliações do teste foram rotuladas em cada tamanho de lote (28 em lotes de 25, de 08/11).

**D31. llama-cpp-python 0.3.36 suporta o Qwen3.5.** Testado em 08/11 com um GGUF **público**
(`unsloth/Qwen3.5-2B-GGUF`, arquivo `Qwen3.5-2B-Q4_K_M.gguf`, baixado do Hugging Face para
`modelos/publico/`, fora do git), sem usar arquivos do Ollama: carregou em 1,9 s e, com o
prefixo `prompt_chat()` do treino e temperatura 0, gerou JSON válido no esquema em 3,3 s
(130 tokens de entrada, 39 de saída), na mesma CPU. A versão fica fixada em `requirements.txt`
e no `requirements.txt` do Space (wheel de CPU de abetlen.github.io). O projetor de visão
(`mmproj`) não é necessário para uso só com texto.

**D32. Pipeline validado de ponta a ponta com o Qwen3.5-0.8B** (rótulos parciais, fora dos
resultados, D28). Kernel `cr-treino-qwen08b-lora` em 08/11: T4, fp16 (a T4 não tem bf16; o
`bf16=False` foi escolhido automaticamente por `torch.cuda.is_bf16_supported()`), 202 exemplos,
3 épocas em 5,4 min, melhor época pela perda de validação, 100% de JSON válido e no esquema nas
20 da validação. Versões: unsloth 2026.11.3, transformers 5.17.0, trl 1.15.0, peft 0.21.2,
torch 2.11.0+cu130. Exportação GGUF q8_0 funcionou (arquivo em `gguf_<nome>_gguf/`).
Conferências:
- o prefixo salvo no Kaggle é **idêntico** a `prompts.prompt_chat()` (usado pela demo e pelo
  Modelfile);
- importado no Ollama local com `ollama/Modelfile` (TEMPLATE explícito), as saídas de 20 exemplos
  da validação tiveram **rótulos idênticos em 20/20** às do Kaggle, e texto idêntico em 15/20
  (só o resumo muda, por causa do q8_0 contra 16 bits);
- o número de tokens do prompt com o meu TEMPLATE é igual ao do `qwen3.5:2b` oficial com o
  renderizador do Ollama e `think: false` (120/120, 160/160, 135/135), ou seja, o 2B base e o 2B
  ajustado recebem o mesmo prompt no Ollama.

**"LoRA bf16" na T4 = LoRA em 16 bits com fp16.** O que importa na recomendação do Unsloth é não
usar 4 bits (QLoRA); a T4 não suporta bf16 em hardware, então o treino usa fp16.

**D33. Intervalos de confiança por bootstrap pareado do teste.** `metricas.bootstrap_f1_macro`
reamostra as 500 avaliações do teste com reposição (1.000 vezes, seed 42) e usa as **mesmas**
reamostragens para todos os modelos, então a diferença entre dois modelos é calculada reamostra a
reamostra (percentis 2,5 e 97,5). Pares reportados: 2B ajustado − 9B e 2B ajustado − TF-IDF; depois
da revisão, também 2B ajustado − Gemini. O IC cobre só a variação de amostragem do teste: não cobre
erro dos rótulos, viés de estilo (D34) nem variação entre treinos (um só seed). Alternativa
descartada: teste de McNemar por categoria, que não dá um intervalo para o F1 macro.

**D34. Antes da revisão, a vantagem do 2B ajustado não é afirmada.** O 2B ajustado imita o estilo
de rotulagem do Gemini e a referência provisória é o Gemini, então a diferença para o 9B e o
TF-IDF mistura qualidade com semelhança de estilo. README, tabela gerada e artigo dizem isso
explicitamente e não afirmam que o 2B supera o 9B até o `recalcular.py` rodar com rótulos humanos.
(Parte da diferença para o 9B é efeito de prompt, não de estilo: ver D37.)

**D35. Checagem de atraso nas duas direções e avaliações feitas antes da entrega.** A tabela
mostra, com números absolutos, quantos atrasados pelas datas cada fonte marca (cobertura) e
quantos dos marcados de fato atrasaram (precisão), para `atraso` e para `atraso` ou
`nao_recebido`. `src/datas_avaliacao.py` mostrou que a maioria dos pedidos atrasados do teste foi
avaliada antes da entrega (`resultados/avaliacao_antes_da_entrega.json`): o texto diz "não
recebi", o que explica a cobertura baixa de `atraso` sozinho. A regra de revisão
(`data/revisao/COMO_REVISAR.md`) manda rotular pelo que o texto diz quando foi escrito. Nada disso
mudou prompts, rótulos ou o modelo.

**D36. Coluna `conferido` na planilha de revisão** (pedido do usuário, 09/11). Sem ela, "coluna
vazia" não distingue "li e aceitei" de "não li". Regras em `recalcular.py`: o teste inteiro precisa
de `x` (senão recusa e diz quantas faltam, sem gravar nada); nos blocos de treino só linhas com `x`
entram; correção sem `x` ou marca diferente de `x` é erro de preenchimento. O README mostra quantas
linhas de cada bloco foram conferidas (`resultados/revisao_humana.json`). A planilha foi regenerada
com a coluna nova; linhas e valores sugeridos são os mesmos de antes.

## Análise conceitual (15/11/2026)

Nenhuma destas decisões retreinou, re-inferiu ou chamou o Kaggle ou o Hugging Face. Os números
antigos das tabelas não mudaram; só entraram linhas e tabelas novas.

**D37. Linha do 9B com o prompt de rotulagem na tabela de qualidade, e releitura da comparação.**
Os baselines "sem ajuste" recebem o prompt curto (`prompt_baseline`), que só lista os nomes das
categorias; as definições e as regras de fronteira (D10) só existem no prompt de rotulagem. O 9B já
tinha rotulado 499 das 500 avaliações do teste com esse prompt, como segundo rotulador
(`data/processed/rotulos_9b.parquet`, lotes de 25, decodificação restrita ao esquema, GPU do
Kaggle), e esse dado não estava na tabela. `src/avaliar.py` passa a avaliá-lo como a linha
"Qwen3.5-9B (prompt de rotulagem, few-shot, lotes de 25) †" (a avaliação que falta conta como sem
previsão) e acrescenta o par 2B ajustado − 9B few-shot ao bootstrap. Resultado, contra os rótulos
do Gemini (`resultados/tabela_qualidade.md`, `resultados/bootstrap_f1_macro.json`): F1 macro
0,819 [0,784; 0,851] e acerto exato 0,788, contra 0,727 do mesmo 9B com o prompt curto e 0,803 do
2B ajustado; diferença 2B ajustado − 9B few-shot de −0,016 [−0,046; +0,016], com IC incluindo
zero (empate). Releitura: a vantagem de +0,076 sobre o "9B sem ajuste" era em boa parte efeito de
prompt, não só de estilo (D34). O que fica de pé, antes da revisão: o 2B ajustado com o prompt
curto (~137 tokens de entrada) empata, em concordância com o Gemini, com um 9B de prompt longo, e
na mesma CPU é ~4,4× mais rápido que o 9B com o prompt **curto** (5,96 s × 26,14 s,
`resultados/tabela_eficiencia.md`); o 9B com o prompt longo não foi medido em CPU e seria mais
lento. Ressalvas: lote, decodificação e hardware da linha † são outros (as colunas de JSON não são
comparáveis), e o empate é contra o Gemini; com rótulos humanos pode mudar, inclusive inverter.
Alternativa: rodar de novo o 9B com o prompt de rotulagem uma avaliação por chamada na CPU local
(≈ 4 h ou mais, D18). Descartada por custo; os rótulos existentes respondem à pergunta conceitual.
Fica em aberto o 2B **sem ajuste** com o prompt de rotulagem, que separaria o efeito do prompt do
efeito do LoRA no próprio 2B: `src/inferir.py` só usa o prompt curto, então exigiria uma opção
nova no script (1 a 1,5 h de CPU local). O dono do projeto optou por não rodar agora.

**D38. Tabela de sensibilidade do F1 macro.** `src/avaliar.py` gera
`resultados/tabela_sensibilidade.md` (e o bloco correspondente no README) com o F1 macro de cada
modelo em três recortes: teste completo; sem as 38 avaliações do teste cujo texto normalizado
(minúsculas, sem acentos e pontuação, `metricas.normalizar_texto`) também aparece no treino
rotulado (n=462); e reponderado para a distribuição de notas da população (o teste tem 62% de
notas 1-3 contra 35% na população com texto, D08; peso = fração na população / fração no teste).
Resultado: a ordem dos modelos não muda em nenhum recorte; sem os textos repetidos o F1 macro cai
no máximo 0,010, e reponderado cai de 0,008 a 0,015 (sobe 0,005 no 2B sem ajuste). Alternativa:
deduplicar o teste contra o treino. Descartada porque o teste está versionado e fixo (D16) e os
textos repetidos ("muito bom") fazem parte da distribuição real; o recorte mostra o efeito sem
mudar a divisão.

**D39. TF-IDF treina com os mesmos rótulos do LoRA.** Antes, o TF-IDF era treinado em
`src/avaliar.py` com a referência do momento: depois da revisão humana, passaria a usar rótulos de
treino corrigidos enquanto o LoRA continuaria treinado com os do Gemini, uma comparação injusta.
Agora ele usa o arquivo que o LoRA usou: o notebook de treino grava `arquivo_rotulos` em
`resultados_qwen2b_lora.json`; sem a chave (caso do treino atual), vale `rotulos_gemini.parquet`.
Se o LoRA for retreinado com `rotulos_revisados.parquet`, o TF-IDF acompanha. A avaliação continua
contra a referência do teste. O arquivo usado fica em `hiperparametros.rotulos_treino` em
`resultados/metricas.json`. Alternativa: retreinar o TF-IDF sempre com a melhor referência
disponível; descartada porque mede outra coisa (TF-IDF com rótulos melhores contra LoRA com
rótulos piores).

**D40. Regra de retreino em `recalcular.py` por limiar.** Antes, bastava um rótulo de treino
mudar para recomendar retreino, o que é sensível demais (retreinar custa fila e GPU no Kaggle e
cerca de 1 h de inferência local). Regra nova, fixada antes de ver os dados da revisão: a taxa de
alteração (sentimento ou conjunto de categorias final diferente do Gemini) é estimada só no bloco
`treino_aleatorio`, nas linhas conferidas, com IC de 95% de Wilson; com menos de 50 conferidas, a
saída é "amostra insuficiente"; com taxa de 10% ou mais, "retreino recomendado"; abaixo disso,
"retreino opcional". Os comandos de retreino são sempre impressos (como opcionais quando a regra
não recomenda). O bloco `treino_discordancia` não entra na taxa (foi escolhido por discordância
e superestima o erro), mas suas correções entram nos rótulos revisados. `linhas_com_correcao`
passa a contar só mudança real de rótulo (preencher a célula com o mesmo valor sugerido não
conta). Alternativa: decidir pelo IC (retreinar se o limite inferior passar de 10%); descartada
porque, com 150 linhas, exigiria uma taxa observada bem acima de 10%.

**D41. Demo com decodificação restrita por gramática e limite de entrada.** `app/app.py` gera uma
gramática GBNF a partir de `esquema.ESQUEMA_SAIDA` (`LlamaGrammar.from_json_schema`) e a usa por
padrão; `CR_SEM_GRAMATICA=1` desliga, e se a criação falhar a demo segue com geração livre. A
entrada é limitada a 1.500 caracteres (o resto é cortado com aviso; `n_ctx=1024`) e erros de
geração viram mensagem em vez de quebrar a interface. Teste local com o GGUF Q4_K_M nos 4
exemplos da demo: mesmos rótulos e resumos com e sem gramática, e 1,5 a 1,7× mais lenta com ela
(ex.: 3,2 s → 5,5 s). A gramática não impõe `uniqueItems` (categorias repetidas continuam
possíveis; a validação na saída marca essa resposta como fora do esquema). Alternativa: só validar a saída, como
antes; mantida a gramática porque um usuário pode colar texto fora da distribuição do treino, e
aí a garantia de formato vale mais que os segundos a mais.

**D42. Latência e custo do Gemini medidos com n=10.** Em 15/11,
`python src/eficiencia.py --gemini --n 10 --nome "Gemini 3.6 Flash (API)"` fez 10 chamadas (as 10
primeiras de `resultados/ids_eficiencia.txt`), uma avaliação por chamada, com o prompt curto, sem
esquema e com o raciocínio no padrão do modelo (`resultados/eficiencia/gemini_3_6_flash__api.jsonl`).
Resultado (`resultados/tabela_eficiencia.md`): média 6,85 s, p95 11,50 s, 100% de JSON válido,
média de 772 tokens de saída cobrados, quase todos de raciocínio (numa chamada, 1.439 de 1.500), e
US$ 2,995 por 1.000 avaliações ao preço atual. O que mostra: nessa amostra pequena, o 2B ajustado
em CPU (5,96 s) teve latência média menor que a API, e o custo do Gemini é dominado pelo
raciocínio, não pela resposta. Ressalvas: n=10, inclui rede, nível gratuito; desligar ou reduzir o
raciocínio do Gemini mudaria latência e custo, e não foi testado. Gastou 10 das 20 requisições do
dia. Alternativa: n maior em vários dias; descartada para não competir com a cota da rotulagem.
Fecha a pendência de D21.

**D43. Tratamento de cota do Gemini unificado.** A nova tentativa ficou em uma função só,
`rotular.chamar_com_retentativa`, usada pela rotulagem e pela inferência/eficiência. Ela lê os
detalhes estruturados do erro 429 (`QuotaFailure.quotaId`, `RetryInfo.retryDelay`) e, na falta
deles, o texto; 429 de cota diária (ou espera sugerida acima de 1 h) levanta
`CotaDiariaEsgotada`, e a inferência para limpo, guardando o que já foi gravado. A rotulagem passa
a gravar os tokens de uso por lote (entrada, saída e raciocínio). Achado: o código antigo só
detectava a cota diária por acaso, porque a regex capturava a mensagem inteira do erro.
Alternativa: manter duas cópias da lógica (rotulagem e inferência); descartada porque já tinham
divergido.
