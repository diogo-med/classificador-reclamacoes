# classificador-reclamacoes

Qwen3.5-2B ajustado com LoRA para classificar avaliações reais de clientes brasileiros (dataset
Olist) e devolver JSON com sentimento, categorias de problema e resumo, comparado a TF-IDF,
Qwen3.5-2B e 9B sem ajuste e ao Gemini 3.6 Flash.

> ⚠️ **Rótulos provisórios.** Todos os rótulos (treino e teste) foram gerados pelo Gemini e a
> revisão humana do teste ainda **não foi feita**. As tabelas abaixo medem quanto cada modelo
> concorda com o Gemini, não com um humano. Elas são regeneradas por `python src/recalcular.py`
> depois da revisão.

**Pergunta central:** medido em rótulos revisados por um humano, quanto da qualidade de um LLM
grande (Gemini) um modelo de 2B ajustado retém, e a que custo e latência?

**Isto é destilação.** Os rótulos de treino vêm do Gemini (o professor) e o 2B (o aluno) aprende
a imitá-lo. Num teste rotulado pelo próprio Gemini, o aluno não tem como superar o professor; por
isso a medida final depende da revisão humana.

```json
{"sentimento": "negativo", "categorias": ["incompleto", "atendimento"], "resumo": "Faltou um item e a loja não responde."}
```

## Resultados

Todos os números abaixo são gerados por `python src/avaliar.py` a partir de arquivos em
`resultados/`.

### Qualidade (teste)
<!-- INICIO:qualidade -->
<!-- FIM:qualidade -->

### F1 por categoria (teste)
<!-- INICIO:f1_categorias -->
<!-- FIM:f1_categorias -->

### Eficiência (mesma CPU, 100 avaliações fixas do teste)
<!-- INICIO:eficiencia -->
<!-- FIM:eficiencia -->

### Concordância Gemini × Qwen3.5-9B (ambiguidade das categorias)
Os dois rotuladores recebem o mesmo prompt de rotulagem. A concordância mede quão ambígua é cada
categoria, não quem está certo.
<!-- INICIO:concordancia -->
<!-- FIM:concordancia -->

### Checagem independente: rótulo `atraso` × atraso real pelas datas
`atraso_real` = entrega depois da data estimada, calculado pelas datas do pedido, sem LLM.
"Cobertura" = fração dos pedidos realmente atrasados que o rótulo marca.
<!-- INICIO:checagem -->
<!-- FIM:checagem -->

## Como funciona
1. `src/preparar_dados.py`: deduplica, calcula o atraso real e filtra avaliações com texto.
2. `src/amostrar.py`: 4.000 avaliações, peso maior nas notas 1-3; teste (500) separado antes de
   qualquer leitura de texto.
3. `src/rotular.py`: rotulagem em lote com o Gemini (e o Qwen3.5-9B como segundo rotulador),
   saída validada por esquema JSON, checkpoint para retomar.
4. `kaggle/treino/notebook.py`: LoRA 16 bits com Unsloth no Kaggle (T4), prompt e chat template
   idênticos aos da inferência, exportação em GGUF.
5. `src/avaliar.py`: métricas de todos os modelos, tabelas e este README.
6. `src/recalcular.py`: aplica a revisão manual e regenera tudo.

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
_Instruções completas quando o GGUF for exportado._
<!-- FIM:ollama -->

## Limitações
- **Domínio:** e-commerce de 2016 a 2018 (Olist), não delivery de comida.
- **Rótulos gerados por LLM** (Gemini 3.6 Flash): o modelo é uma destilação e herda os erros do
  professor.
- **Revisão manual pendente**; quando feita, foi feita vendo o rótulo sugerido (**viés de
  ancoragem**: a concordância tende a sair superestimada).
- **Cota gratuita do Gemini** (20 requisições/dia): lotes de 100 avaliações, e o baseline do
  Gemini com o prompt curto não foi executado (ver DECISOES.md D19-D21).
- Um só seed, uma só divisão; categorias raras têm poucas dezenas de exemplos no teste.

## Licença
Dados derivados do [Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce),
CC BY-NC-SA 4.0. Modelo e dataset derivados seguem a mesma licença.
