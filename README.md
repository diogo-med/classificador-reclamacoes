# classificador-reclamacoes

Qwen3.5-2B ajustado com LoRA para classificar avaliações reais de clientes brasileiros (dataset
Olist) e devolver JSON com sentimento, categorias de problema e resumo, comparado a TF-IDF,
Qwen3.5-2B e 9B sem ajuste e ao Gemini 3.6 Flash.

<!-- INICIO:status -->
> ⚠️ **Rótulos provisórios.** Todos os rótulos (treino e teste) foram gerados pelo Gemini e a
> revisão humana do teste ainda **não foi feita**. As tabelas abaixo medem quanto cada modelo
> concorda com o Gemini, não com um humano. Elas são regeneradas por `python src/recalcular.py`
> depois da revisão.
<!-- FIM:status -->

**Pergunta central:** medido em rótulos revisados por um humano, quanto da qualidade de um LLM
grande (Gemini) um modelo de 2B ajustado retém, e a que custo e latência?

**Isto é destilação.** Os rótulos de treino vêm do Gemini (o professor) e o 2B (o aluno) aprende
a imitá-lo. Num teste rotulado pelo próprio Gemini, o aluno não tem como superar o professor; por
isso a medida final depende da revisão humana.

**Leitura dos resultados antes da revisão: a vantagem do 2B ajustado está inflada.** O 2B
ajustado foi treinado para imitar o estilo de rotulagem do Gemini (quais categorias marcar
juntas, onde fica a fronteira de `atendimento`, `outro` etc.), e as tabelas abaixo medem a
concordância com o mesmo Gemini. O Qwen3.5-9B e o 2B sem ajuste não viram esse estilo, e o
TF-IDF viu os mesmos rótulos com muito menos capacidade. Por isso a diferença a favor do 2B
ajustado mistura qualidade com semelhança de estilo. **Este README não afirma que o 2B ajustado
supera o 9B ou o TF-IDF**; essa comparação só vale depois de `python src/recalcular.py` rodar com
os rótulos revisados por humano, quando as tabelas e os intervalos de confiança são refeitos.

**Parte da vantagem sobre o "9B sem ajuste" é efeito de prompt.** Os baselines "sem ajuste"
recebem o prompt curto, que só lista os nomes das categorias; as definições e regras (por exemplo,
`atraso` × `nao_recebido`) só existem no prompt de rotulagem. O mesmo Qwen3.5-9B com o prompt de
rotulagem (linha † da tabela de qualidade) chega a F1 macro 0,819 [0,784; 0,851] em concordância
com o Gemini, contra 0,727 com o prompt curto e 0,803 do 2B ajustado. A diferença 2B ajustado − 9B
few-shot é −0,016 [−0,046; +0,016]: o intervalo inclui zero, ou seja, empate
(`resultados/tabela_qualidade.md`, `resultados/bootstrap_f1_macro.json`). A leitura honesta, por
enquanto: o 2B ajustado, com um prompt curto (~137 tokens de entrada), empata em concordância com
o Gemini com um 9B de prompt longo, e na mesma CPU leva 5,96 s por avaliação contra 26,14 s do 9B
com o prompt **curto** (~4,4× mais rápido; o 9B com o prompt longo não foi medido em CPU e seria
mais lento; `resultados/tabela_eficiencia.md`). A linha † rodou em lotes de 25, com decodificação
restrita ao esquema, na GPU do Kaggle; e o empate também é medido contra o Gemini, então pode
mudar (inclusive inverter) com os rótulos humanos.

```json
{"sentimento": "negativo", "categorias": ["incompleto", "atendimento"], "resumo": "Faltou um item e a loja não responde."}
```

## Modelo, dataset e demo
No Hugging Face, com os rótulos **provisórios** do Gemini (serão atualizados depois da revisão
humana):
- Modelo (adaptador LoRA e GGUFs Q8_0 e Q4_K_M): [Diogv20/classificador-reclamacoes-qwen3.5-2b](https://huggingface.co/Diogv20/classificador-reclamacoes-qwen3.5-2b)
- Dataset (4.000 avaliações, 2.050 rotuladas): [Diogv20/classificador-reclamacoes-olist](https://huggingface.co/datasets/Diogv20/classificador-reclamacoes-olist)
- Demo (Gradio, CPU): [Diogv20/classificador-reclamacoes-demo](https://huggingface.co/spaces/Diogv20/classificador-reclamacoes-demo)

## Resultados

Todos os números abaixo são gerados por `python src/avaliar.py` a partir de arquivos em
`resultados/`.

### Revisão humana
<!-- INICIO:revisao -->
Revisão humana **ainda não feita**: nenhuma linha conferida. Os rótulos de referência são os do Gemini.
<!-- FIM:revisao -->

### Qualidade (teste)
<!-- INICIO:qualidade -->
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
<!-- FIM:qualidade -->

### F1 por categoria (teste)
<!-- INICIO:f1_categorias -->
| Modelo | `atraso` | `nao_recebido` | `incompleto` | `produto_errado` | `defeito_qualidade` | `atendimento` | `elogio` | `outro` |
|---|---|---|---|---|---|---|---|---|
| TF-IDF + regressão logística | 0.69 | 0.77 | 0.72 | 0.71 | 0.65 | 0.66 | 0.93 | 0.33 |
| Qwen3.5-2B sem ajuste (Ollama q8_0) | 0.38 | 0.71 | 0.33 | 0.59 | 0.76 | 0.26 | 0.90 | 0.27 |
| Qwen3.5-9B sem ajuste (Ollama Q4_K_M) | 0.57 | 0.84 | 0.77 | 0.86 | 0.85 | 0.65 | 0.96 | 0.32 |
| Qwen3.5-9B (prompt de rotulagem, few-shot, lotes de 25) | 0.72 | 0.90 | 0.92 | 0.91 | 0.84 | 0.74 | 0.98 | 0.53 |
| Qwen3.5-2B + LoRA (este projeto) | 0.81 | 0.89 | 0.82 | 0.85 | 0.85 | 0.74 | 0.96 | 0.50 |
| Qwen3.5-2B + LoRA, GGUF q8_0 no Ollama | 0.82 | 0.90 | 0.82 | 0.85 | 0.85 | 0.75 | 0.96 | 0.52 |
| *suporte no teste* | 50 | 92 | 40 | 67 | 66 | 76 | 187 | 35 |
<!-- FIM:f1_categorias -->

### Sensibilidade (teste)
<!-- INICIO:sensibilidade -->
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
<!-- FIM:sensibilidade -->

### Eficiência (mesma CPU, 100 avaliações fixas do teste)
<!-- INICIO:eficiencia -->
Hardware local: 12th Gen Intel(R) Core(TM) i5-1235U, 12 CPUs lógicas, 15Gi de RAM, sem GPU dedicada (só CPU).

| Modelo | Onde | n | Latência média (s) | Latência p95 (s) | Tokens/s (geração) | JSON válido (%) | Custo por 1.000 avaliações |
|---|---|---|---|---|---|---|---|
| Qwen3.5-2B sem ajuste (q8_0) | Ollama local (CPU) | 100 | 7.21 | 10.02 | 10.9 | 100.0 | US$ 0 de API (hardware próprio) |
| Qwen3.5-9B sem ajuste (Q4_K_M) | Ollama local (CPU) | 100 | 26.14 | 36.37 | 3.4 | 100.0 | US$ 0 de API (hardware próprio) |
| Qwen3.5-2B + LoRA (q8_0) | Ollama local (CPU) | 100 | 5.96 | 7.27 | 11.2 | 100.0 | US$ 0 de API (hardware próprio) |
| Qwen3.5-2B + LoRA (q8_0, format=esquema) | Ollama local (CPU) | 100 | 6.07 | 7.62 | 10.8 | 100.0 | US$ 0 de API (hardware próprio) |
| Gemini 3.6 Flash (API) | API (rede) | 10 | 6.85 | 11.50 | — | 100.0 | US$ 2.995 |

- Mesma amostra fixa de 100 avaliações do teste para os modelos locais (`resultados/ids_eficiencia.txt`), uma chamada por avaliação, sem paralelismo, temperatura 0, raciocínio desligado, num_ctx 2048, 1 chamada de aquecimento descartada.
- Latência = tempo de parede da requisição HTTP ao Ollama local (inclui processar o prompt e gerar a resposta).
- Modelos locais: custo marginal de API = 0; o custo real é o hardware/energia, que não estimamos.
- Gemini 3.6 Flash (API): latência da API medida em 10 chamadas (nível gratuito, limitado a 20 requisições/dia), inclui rede; não é comparável diretamente à CPU local.
- Preço gemini-3.6-flash: US$ 0.75/1M tokens de entrada e US$ 3.75/1M de saída (https://aistudio.google.com/docs/pricing (redirecionado de ai.google.dev/gemini-api/docs/pricing), nível pago Standard, texto, consultado em 2026-11-08). Preço promocional válido até 31/12/2026; a partir de 01/01/2027 passa a US$ 1,50 (entrada) e US$ 7,50 (saída). A saída inclui tokens de raciocínio. Batch API: metade do preço. O nível gratuito não cobra, mas é limitado a 20 requisições/dia.
<!-- FIM:eficiencia -->

### Concordância Gemini × Qwen3.5-9B (ambiguidade das categorias)
Os dois rotuladores recebem o mesmo prompt de rotulagem. A concordância mede quão ambígua é cada
categoria, não quem está certo.
<!-- INICIO:concordancia -->
| Recorte | n | Sentimento (% / κ) | `atraso` (% / κ) | `nao_recebido` (% / κ) | `incompleto` (% / κ) | `produto_errado` (% / κ) | `defeito_qualidade` (% / κ) | `atendimento` (% / κ) | `elogio` (% / κ) | `outro` (% / κ) | Conjunto idêntico (%) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| teste | 499 | 94 / 0.90 | 92 / 0.68 | 97 / 0.89 | 99 / 0.91 | 98 / 0.90 | 96 / 0.82 | 92 / 0.70 | 99 / 0.97 | 94 / 0.50 | 79.0 |
| treino | 378 | 94 / 0.89 | 95 / 0.81 | 98 / 0.92 | 97 / 0.87 | 97 / 0.80 | 97 / 0.88 | 94 / 0.75 | 98 / 0.95 | 94 / 0.60 | 79.4 |
<!-- FIM:concordancia -->

### Checagem independente: rótulo `atraso` × atraso real pelas datas
`atraso_real` = entrega depois da data estimada, calculado pelas datas do pedido, sem LLM.
"Cobertura" = fração dos pedidos realmente atrasados que o rótulo marca.
<!-- INICIO:checagem -->
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
<!-- FIM:checagem -->

## Como funciona
1. `src/preparar_dados.py`: deduplica, calcula o atraso real e filtra avaliações com texto.
2. `src/amostrar.py`: 4.000 avaliações, peso maior nas notas 1-3; teste (500) separado antes de
   qualquer leitura de texto.
3. `src/rotular.py`: rotulagem em lote com o Gemini (e o Qwen3.5-9B como segundo rotulador),
   saída validada por esquema JSON, checkpoint para retomar.
4. `kaggle/treino/notebook.py`: LoRA 16 bits com Unsloth no Kaggle (T4), prompt e chat template
   idênticos aos da inferência, exportação em GGUF.
5. `src/avaliar.py`: métricas de todos os modelos, tabelas e este README (inclusive o aviso de
   status no topo e a tabela de sensibilidade). O TF-IDF é treinado aqui com os mesmos rótulos
   com que o LoRA foi treinado (o notebook grava `arquivo_rotulos`; sem a chave, Gemini).
6. `src/recalcular.py`: aplica a revisão manual, regenera tudo e decide se vale retreinar: estima
   a taxa de rótulos alterados no bloco `treino_aleatorio` (IC de Wilson) e só recomenda retreino
   com pelo menos 50 linhas conferidas e taxa de 10% ou mais (regra fixada antes da revisão).
7. `app/app.py`: demo em Gradio com llama-cpp-python, decodificação restrita por uma gramática
   gerada do esquema (`CR_SEM_GRAMATICA=1` desliga) e entrada limitada a 1.500 caracteres.

Decisões e alternativas: [DECISOES.md](DECISOES.md). Justificativas e perguntas de entrevista:
[ESTUDO.md](ESTUDO.md). Rascunho do artigo: [docs/artigo.md](docs/artigo.md).

## Como reproduzir
```bash
conda create -n classificador-reclamacoes python=3.11 -y && conda activate classificador-reclamacoes
pip install -r requirements.txt
cd data/raw && kaggle datasets download olistbr/brazilian-ecommerce --unzip && cd ../..
python src/preparar_dados.py && python src/amostrar.py
python src/avaliar.py          # regenera resultados/ a partir das previsões versionadas
pytest
```
A rotulagem precisa de `GEMINI_API_KEY` no `.env` (ver `.env.example`). O treino roda no Kaggle:
`python kaggle/gerar_notebooks.py && kaggle kernels push -p kaggle/treino_qwen2b_lora`.

## Rodar localmente com Ollama
<!-- INICIO:ollama -->
O GGUF sai do notebook de treino no Kaggle (`outputs/kaggle/treino_qwen2b_lora/`) e fica fora do
git. Copie para `modelos/` e crie o modelo no Ollama com o `ollama/Modelfile`, cujo TEMPLATE
reproduz o prefixo usado no treino (raciocínio desligado, temperatura 0):

```bash
cp outputs/kaggle/treino_qwen2b_lora/gguf_qwen2b_lora_gguf/Qwen3.5-2B.Q8_0.gguf modelos/qwen2b_lora-Q8_0.gguf
ollama create cr-qwen2b-lora -f ollama/Modelfile
```

O modelo espera o mesmo prompt do treino (`src/prompts.py`, `prompt_baseline`):

```bash
python src/inferir.py --backend ollama --modelo cr-qwen2b-lora --split teste --n 5 --saida outputs/exemplo.jsonl
python - <<'PY'
import sys; sys.path.insert(0, "src")
from prompts import prompt_baseline
from rotular import chamar_ollama
print(chamar_ollama(prompt_baseline("Veio só um dos dois itens e a loja não responde."), "cr-qwen2b-lora")[0])
PY
```

Conferência Ollama (q8_0) × Kaggle (16 bits) em 20 avaliações da validação:
`resultados/conferencia_ollama_kaggle.json`. Para a demo sem Ollama (llama-cpp-python + Gradio,
GGUF Q4_K_M): `python app/app.py`.
<!-- FIM:ollama -->

## Limitações
- **Domínio:** e-commerce de 2016 a 2018 (Olist), não delivery de comida.
- **Rótulos gerados por LLM** (Gemini 3.6 Flash): o modelo é uma destilação e herda os erros do
  professor.
- **Revisão manual**: o aviso no topo diz se ela já foi feita. Enquanto não for, a referência é
  o Gemini. Quando for, terá sido feita vendo o rótulo sugerido (**viés de ancoragem**: a
  concordância tende a sair superestimada); uma subamostra revisada às cegas mediria esse viés, e
  não está prevista.
- **Vantagem do 2B ajustado inflada até a revisão**: ele imita o estilo do Gemini, e a referência
  provisória é o próprio Gemini (ver "Leitura dos resultados" acima). A revisão com o rótulo
  sugerido à vista (ancoragem) puxa na mesma direção, então mesmo depois dela a vantagem pode
  continuar um pouco superestimada.
- **Assimetria de prompt nos baselines**: os baselines "sem ajuste" usam o prompt curto, sem
  definições. A linha † (9B com o prompt de rotulagem) mostra que isso pesa: ela empata com o 2B
  ajustado. Falta o 2B **sem ajuste** com o prompt de rotulagem completo, que separaria, no próprio
  2B, o efeito do prompt do efeito do LoRA; `src/inferir.py` hoje só usa o prompt curto, então
  esse experimento exige uma opção nova no script (estimativa: 1 a 1,5 h de CPU local).
- **Textos repetidos entre treino e teste**: 38 avaliações do teste têm texto normalizado igual
  ao de alguma do treino rotulado (em geral textos curtos como "muito bom"). Sem elas (n=462), o
  F1 macro cai no máximo 0,010 e a ordem dos modelos não muda (tabela de sensibilidade).
- **Distribuição de notas**: o teste tem 62% de notas 1-3, contra 35% na população com texto
  (D08). Reponderado para a população, o F1 macro cai de 0,008 a 0,015 (sobe 0,005 no 2B sem
  ajuste), e a ordem não muda (tabela de sensibilidade).
- **Latência do Gemini com n=10**: medida em 10 chamadas pela rede, no nível gratuito, com o
  raciocínio no padrão do modelo (que domina os tokens de saída cobrados); desligar ou reduzir o
  raciocínio mudaria latência e custo, e não foi testado.
- **Treino parcial**: só 1.200 das 3.150 avaliações de treino têm rótulo; rotular o resto
  custaria um dia de cota do Gemini e exigiria retreino.
- **Intervalos de confiança** (bootstrap do teste) cobrem só a variação de amostragem das 500
  avaliações, não o erro dos rótulos nem a variação entre treinos.
- **Cota gratuita do Gemini** (20 requisições/dia): lotes de 100 avaliações, e o baseline do
  Gemini com o prompt curto não foi executado (ver DECISOES.md D19-D21).
- Um só seed, uma só divisão; categorias raras têm poucas dezenas de exemplos no teste.

## Licença
Dados derivados do [Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce),
CC BY-NC-SA 4.0. Modelo e dataset derivados seguem a mesma licença.
