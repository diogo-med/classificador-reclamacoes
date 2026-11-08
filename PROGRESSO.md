# Progresso

Atualizado ao fim de cada etapa. Se a sessão cair, retomar a partir daqui.

## Estado em 08/11/2026 (retomada após a queda do VS Code)

### Feito
- **Etapa 1 (parcial):** estrutura de pastas, `.gitignore`, `.env.example`, `data/README.md`,
  `docs/plano.md` (renomeado), `DECISOES.md`, `src/config.py` (caminhos, seed 42).
  Falta: `requirements.txt` fixado, `tests/`, `ESTUDO.md`, README completo.
- **Etapa 2:** `src/preparar_dados.py` gera `data/interim/avaliacoes_com_texto.parquet` (40.312
  avaliações com texto) e `resultados/estatisticas_dados.json`.
- **Etapa 3:** `src/amostrar.py` gera `data/processed/amostra.parquet`: 4.000 avaliações,
  treino 3.150 / validação 350 / teste 500, estratificado por nota (`resultados/amostragem.json`).
- **Etapa 4:** 200 avaliações do treino lidas; 8 categorias definidas em `src/prompts.py`
  (`atendimento` no lugar de `sem_resposta`; ver D09-D13).
- **Etapa 5 (em andamento):** `src/rotular.py` + `src/esquema.py` + `PROMPT_ROTULAGEM` v1.
  Modelo `gemini-3.6-flash`. Checkpoint em `outputs/rotulos_gemini.jsonl` (fora do git).
  Antes da queda: 9 lotes = 225 avaliações rotuladas, todas válidas no esquema. A última rodada
  parou num 429 que o código tratou como cota diária; uma chamada logo depois passou, então o log
  agora registra a métrica exata da cota. Rotulagem retomada em segundo plano.
- `src/inferir.py` (baselines LLM, uma avaliação por chamada) escrito. Medição local do
  qwen3.5:2b: 9,6 s/avaliação (ver tabela em DECISOES.md). Medição do 9B morreu por falta de
  memória (D18).

### Falta
Etapas 5 (concluir) a 16.

### Próximos passos
1. Terminar a rotulagem do Gemini; consolidar em `data/processed/rotulos_gemini.parquet`.
2. Criar o dataset privado no Kaggle e o kernel do Qwen3.5-9B (segundo rotulador + baseline).
3. Baseline 2B local no teste; TF-IDF; Gemini baseline.
4. Notebook de treino (0.8B, depois 2B).

### Para retomar
```bash
conda activate classificador-reclamacoes
python -u src/rotular.py --rotulador gemini --saida outputs/rotulos_gemini.jsonl --pausa 4
```
(pula os lotes já feitos)
