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
reporta também `atraso OU nao_recebido`.

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
nos rótulos humanos. A latência do Gemini usa poucas chamadas avulsas (cota), com n declarado.

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
