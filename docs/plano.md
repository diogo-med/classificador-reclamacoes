# Classificador de reclamações em PT-BR (Olist) — base do projeto

Nov 2, 2026 · @Diogo

## Visão geral

O projeto, chamado classificador-reclamacoes, treina um modelo de linguagem pequeno, com LoRA, para classificar reclamações reais de clientes brasileiros e devolver o resultado em JSON. A entrega final é um repositório no GitHub, um modelo e um dataset publicados no Hugging Face, uma demo em Space e um notebook no Kaggle.

O foco é a trilha AI Engineer do GenAI iFood, cujo portfólio esperado inclui implementações de ML no GitHub, modelos no Hugging Face, Kaggle e artigos técnicos. O projeto cobre os quatro itens de uma vez.

Pergunta central do projeto: um modelo pequeno ajustado consegue igualar ou superar um LLM grande usado só com prompt, a um custo e latência bem menores?

## Dataset: Olist Brazilian E-Commerce

O dataset tem cerca de 100 mil pedidos reais de marketplaces brasileiros entre 2016 e 2018, divididos em 9 CSVs ligados por `order_id`. Para este projeto importam só dois arquivos.

| Arquivo | Linhas | Colunas usadas |
| --- | --- | --- |
| `olist_order_reviews_dataset.csv` | 99.224 | `review_id`, `order_id`, `review_score` (1–5), `review_comment_title`, `review_comment_message`, `review_answer_timestamp` |
| `olist_orders_dataset.csv` | 99.441 | `order_id`, `order_status`, `order_delivered_customer_date`, `order_estimated_delivery_date` |

Pontos de atenção na limpeza:

- Cerca de 59% das avaliações não têm texto; sobram por volta de 41 mil comentários utilizáveis.
- 551 pedidos têm mais de uma avaliação; manter a mais recente por `review_answer_timestamp`.
- As datas vêm como texto; converter com `pd.to_datetime`.
- Filtrar `order_status = delivered` para calcular atraso; cerca de 6,6% dos pedidos entregues chegaram depois do prazo estimado.

Pergunta aberta: confirmar a licença na página do Kaggle antes de republicar qualquer subconjunto no Hugging Face. Se ela for não comercial ou exigir atribuição, repetir os mesmos termos no dataset card.

## Definição da tarefa

O modelo recebe o texto de uma avaliação e devolve um JSON com o sentimento e as categorias de problema citadas. Uma avaliação pode citar mais de uma categoria (classificação multirrótulo).

Saída esperada:

```json
{"sentimento": "negativo", "categorias": ["atraso", "sem_resposta"], "resumo": "Pedido chegou 10 dias depois e o vendedor não respondeu"}
```

Categorias propostas (ajustar depois de ler umas 200 avaliações):

| Categoria | Exemplo do que cobre |
| --- | --- |
| `atraso` | entregou depois do prazo, ainda aguardando |
| `nao_recebido` | pedido nunca chegou, consta entregue mas não chegou |
| `incompleto` | faltou item ou chegou só parte do pedido |
| `produto_errado` | cor, modelo ou produto diferente do comprado |
| `defeito_qualidade` | quebrado, com defeito, qualidade abaixo do esperado |
| `sem_resposta` | vendedor ou loja não respondeu, dificuldade de troca |
| `elogio` | avaliação positiva, sem problema |
| `outro` | não se encaixa nas anteriores |

Como rotular sem inventar dados:

1. Amostrar de 3 a 5 mil avaliações com texto, com mais peso para notas 1 a 3, onde estão as reclamações.
2. Rotular essa amostra com a API do Gemini (nível gratuito, modelo Flash), enviando de 20 a 30 avaliações por requisição, com instruções e exemplos fixos.
3. Revisar à mão umas 300 avaliações sorteadas e medir a concordância com o LLM.
4. Separar um conjunto de teste de \~500 exemplos revisados à mão, que nunca entra no treino.
5. Usar as datas como verificação independente: pedidos com entrega real depois da estimada devem concentrar a categoria `atraso`.

O passo 5 é o diferencial: um rótulo que vem dos dados, e não de um LLM, para checar tanto a rotulagem quanto o modelo final.

## Modelo e técnica

A escolha é Qwen3.5-2B com LoRA em bf16 via Unsloth, que segundo a documentação usa cerca de 5 GB de VRAM e tem notebook gratuito no Colab. Se o treino ficar lento ou instável, a alternativa é Qwen3-1.7B, só texto e mais madura.

| Opção | VRAM (LoRA bf16) | Quando usar |
| --- | --- | --- |
| Qwen3.5-0.8B | \~3 GB | teste rápido do pipeline |
| Qwen3.5-2B | \~5 GB | versão principal |
| Qwen3.5-4B | \~10 GB | se sobrar tempo, para comparar tamanho × qualidade |

Cuidados que a documentação do Unsloth aponta para Qwen3.5:

- Não usar QLoRA (4 bits) nesses modelos; a perda por quantização é maior que o normal.
- O Qwen3.5 é um modelo multimodal e compila kernels próprios, então o primeiro treino em GPU T4 demora mais.
- Usar o mesmo chat template no treino e na inferência; template errado é a causa mais comum de modelo exportado se comportar mal.

Configuração inicial sugerida: rank `r = 16`, `lora_alpha = 16`, `max_seq_length = 512` (avaliações são curtas), 2 a 3 épocas, `SFTTrainer` do TRL.

Bibliotecas: `unsloth`, `transformers`, `trl`, `peft`, `datasets`, `pandas`, `scikit-learn` (métricas).

Extra opcional, se sobrar tempo: treinar também um classificador clássico (TF-IDF + regressão logística) como piso de comparação. Leva menos de uma hora e mostra que você mediu se o LLM era mesmo necessário.

## Ambiente de treino gratuito

O treino roda no Kaggle Notebooks, disparado pela CLI do Kaggle a partir do Claude Code. A máquina local (Linux, GPU integrada Intel Iris Xe) não tem GPU NVIDIA com CUDA, que o Unsloth exige, então fica só com preparação de dados, rotulagem, baselines leves e a demo em CPU. A cota semanal de GPU do Kaggle aparece na conta e é relatada em torno de 30 horas; o Colab gratuito fica como reserva.

| Plataforma | GPU | Limite de sessão | Cota | Uso no projeto |
| --- | --- | --- | --- | --- |
| Kaggle Notebooks | T4 ou P100 | 12 h | semanal, visível na conta (\~30 h relatadas) | treino principal e notebook público |
| Google Colab (gratuito) | não informada (usuários relatam T4) | até 12 h | não publicada | testes rápidos e notebooks do Unsloth |
| Hugging Face ZeroGPU | RTX PRO 6000 compartilhada | funções de 60 s | 5 min de GPU por dia na conta gratuita | só demo, não treina |

Na prática, com \~4 mil exemplos curtos e um modelo de 2B, cada treino deve levar bem menos que uma sessão. Isso deixa cota para 4 a 6 rodadas de ajuste.

Cuidados:

- O notebook do Kaggle não recebe nenhuma chave: salva adaptador LoRA, previsões e métricas em `/kaggle/working`, e o Claude Code baixa tudo com `kaggle kernels output`.
- Os dados rotulados sobem para o Kaggle como dataset privado seu e entram como fonte do notebook.
- Manter o notebook privado (`is_private: true`) até a revisão.

## Execução com Claude Code no Linux

O Claude Code executa as etapas sozinho a partir do prompt e registra o andamento em `PROGRESSO.md` e as decisões em `DECISOES.md`. Antes de iniciar, deixe pronto:

1. Pasta do projeto (clone do repositório privado `classificador-reclamacoes`) com este documento exportado em `docs/plano.md`.
2. Ambiente conda dedicado ao projeto: `conda create -n classificador-reclamacoes python=3.11 -y && conda activate classificador-reclamacoes`.
3. `.env` só com `GEMINI_API_KEY` (gerada no Google AI Studio); os scripts carregam com `python-dotenv`.
4. Token novo do Kaggle (seção "API Tokens", não a legada) salvo em `~/.kaggle/access_token`, com `chmod 600`, e CLI `kaggle` 1.8.0 ou mais recente. Conta verificada por telefone para liberar GPU.
5. Repositório configurado como `origin`, com `git push` funcionando no terminal.
6. `.claude/settings.json` liberando `python`, `pip`, `kaggle`, `git add`, `git commit` e `git push`, e negando leitura do `.env`.

O token do Hugging Face não entra agora: ele só é usado na publicação, depois da revisão.

## Avaliação

Todos os modelos são comparados no mesmo conjunto de teste de \~500 avaliações revisadas à mão. O resultado vira a tabela principal do README.

| Modelo | F1 macro | F1 `atraso` | JSON válido (%) | Latência média | Custo por 1.000 avaliações |
| --- | --- | --- | --- | --- | --- |
| TF-IDF + regressão logística |  |  |  |  |  |
| Qwen3.5-2B sem ajuste (só prompt) |  |  |  |  |  |
| Gemini via API (só prompt) |  |  |  |  |  |
| Qwen3.5-2B + LoRA (este projeto) |  |  |  |  |  |

O que medir e por quê:

- **F1 macro:** média entre as categorias, para a classe rara não sumir atrás das frequentes.
- **F1 por categoria:** mostra onde o modelo erra; publicar também a matriz de confusão.
- **JSON válido:** um modelo que acerta mas quebra o formato não serve em produção.
- **Latência e custo:** o argumento central a favor de um modelo pequeno.
- **Checagem com as datas:** entre os pedidos realmente atrasados que têm comentário, que fração o modelo marca como `atraso`.

Incluir no README uma seção curta de análise de erros, com 5 a 10 exemplos onde o modelo errou e uma hipótese para cada. Isso mostra maturidade mais do que um número alto.

## Publicação

Cada entrega cobre um item do portfólio que o iFood pede para a trilha AI Engineer, e todas apontam umas para as outras.

Ordem da publicação: durante o desenvolvimento, o Claude Code só faz push para o repositório privado no GitHub e mantém o notebook do Kaggle privado. Depois da revisão, você roda `src/recalcular.py`, publica no Hugging Face com `src/publicar.py` e o seu `HF_TOKEN`, e torna o repositório e o notebook públicos.

| Onde | O que publicar | Item do edital que cobre |
| --- | --- | --- |
| GitHub | código, README com resultados e análise de erros | GitHub com implementações de ML |
| Hugging Face (modelo) | adaptador LoRA e/ou modelo mesclado, com model card | modelos publicados no Hugging Face |
| Hugging Face (dataset) | conjunto rotulado e de teste, com dataset card | idem |
| Hugging Face (Space) | demo em Gradio: cola uma avaliação, recebe o JSON | demo funcional |
| Kaggle | notebook público de exploração e treino com o Olist | participações no Kaggle |
| dev.to, Medium ou LinkedIn | artigo técnico com decisões e números | artigos técnicos |

Model card: é o `README.md` do repositório do modelo, com um bloco YAML no topo. Preencher pelo menos `base_model` (o Qwen usado), `datasets` (o seu dataset no Hub), `language: pt`, `license` e os resultados de avaliação, que o Hub mostra num widget na página. Como o repositório é novo, declarar `library_name` explicitamente.

Dataset card: descrever a origem (Olist), como os rótulos foram gerados, a taxa de concordância da revisão manual e as limitações (e-commerce de 2016–2018, não delivery de comida).

Demo: a ZeroGPU gratuita só dá 5 minutos de GPU por dia. Para uma demo sempre disponível, exportar o modelo em GGUF quantizado (o Unsloth faz isso direto) e rodar em CPU no Space gratuito.

## Estrutura do repositório

Separar notebooks (exploração) de código reutilizável (`src/`), para que o treino e a avaliação possam rodar por linha de comando.

```text
classificador-reclamacoes/
├── README.md              # problema, resultados, como rodar, análise de erros
├── PROGRESSO.md           # andamento e resumo final do Claude Code
├── DECISOES.md            # decisões tomadas sem consulta, com motivo
├── requirements.txt
├── .env.example           # nomes das variáveis, sem chaves
├── .claude/settings.json  # permissões do Claude Code
├── docs/
│   ├── plano.md           # este documento
│   └── artigo.md          # rascunho do artigo técnico
├── data/
│   ├── README.md          # como baixar o Olist (CSVs fora do git)
│   └── revisao/
│       └── revisao_manual.csv
├── notebooks/
│   ├── 01_exploracao.ipynb
│   └── 02_rotulagem.ipynb
├── kaggle/
│   ├── treino.ipynb       # notebook enviado ao Kaggle
│   └── kernel-metadata.json
├── src/
│   ├── preparar_dados.py  # limpeza, deduplicação, cálculo de atraso
│   ├── rotular.py         # rotulagem em lotes com a API do Gemini
│   ├── prompts.py         # prompts versionados
│   ├── treinar.py         # LoRA com Unsloth + TRL
│   ├── avaliar.py         # métricas e tabela comparativa
│   ├── recalcular.py      # reaplica a revisão manual e regenera resultados
│   └── publicar.py        # envia modelo, dataset e Space ao Hugging Face
├── app/
│   └── app.py             # demo Gradio com GGUF em CPU
└── resultados/
    ├── metricas.json
    └── matriz_confusao.png
```

README, nesta ordem: uma frase com o resultado principal, a tabela de métricas, links para modelo, dataset, Space e notebook, como reproduzir, decisões e limitações.

## Cronograma até 12/nov

Cronograma refeito em 8/nov: o Claude Code executa o projeto de forma autônoma e você entra na configuração, na revisão manual e na publicação. A inscrição continua prevista para 12/nov, com folga até o fim das inscrições em 17/nov. Se atrasar, cortar primeiro o modelo de 4B e o artigo longo, nunca a avaliação.

- [ ] **8/nov:** criar contas e chaves, preparar a pasta e iniciar o Claude Code
- [ ] **9/nov:** Claude Code roda dados, rotulagem, baselines e treino; acompanhar o PROGRESSO.md
- [ ] **10/nov:** revisar à mão a planilha de rotulagem (no mínimo o conjunto de teste) e rodar o recalcular.py
- [ ] **11/nov:** revisar README, model card e dataset card; tornar tudo público; ajustar o rascunho do artigo
- [ ] **12/nov:** inscrição no GenAI iFood com os links no currículo

## Fontes

Páginas consultadas entre 2 e 8/nov/2026.

- [Brazilian E-Commerce Public Dataset by Olist (Kaggle)](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce): página oficial do dataset; conferir a licença aqui
- [Why Customers Never Return: The Dark Side of Olist's Logistics (AIO Conquer)](https://aioconquer.aivietnam.edu.vn/posts/why-customers-never-return-the-dark-side-of-olist-s-logistics): contagem de linhas, valores ausentes, avaliações duplicadas e taxa de atraso
- [Qwen3.5 Fine-tuning Guide (Unsloth)](https://unsloth.ai/docs/models/qwen3.5/fine-tune.md): tamanhos de modelo, VRAM por tamanho, notebooks gratuitos, aviso sobre QLoRA, exportação para GGUF e Hub
- [Qwen3 - How to Run & Fine-tune (Unsloth)](https://unsloth.ai/docs/models/tutorials/qwen3-how-to-run-and-fine-tune): alternativa Qwen3 só texto
- [Free Cloud GPUs in 2026 (GPUPerHour)](https://gpuperhour.com/blog/free-cloud-gpus-and-credits): limites de Kaggle, Colab e ZeroGPU
- [Model Cards (Hugging Face Hub docs)](https://huggingface.co/docs/hub/en/model-cards): metadados do model card, `base_model`, `datasets`, `library_name`, resultados de avaliação
- [Repository Cards (huggingface\_hub)](https://huggingface.co/docs/huggingface_hub/package_reference/cards): criar model e dataset cards por código

* [Configure permissions (Claude Code docs)](https://code.claude.com/docs/en/permissions): regras de `allow` e `deny`, opção "Yes, don't ask again" e comando `/permissions`

- [Kaggle CLI Documentation](https://raw.githubusercontent.com/Kaggle/kaggle-api/main/docs/README.md): autenticação com o token novo em `~/.kaggle/access_token` ou na variável `KAGGLE_API_TOKEN`
