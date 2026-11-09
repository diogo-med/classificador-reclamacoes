# Progresso

Atualizado ao fim de cada etapa. Se a sessão cair, retomar a partir daqui.

## Estado em 08/11/2026, 21:45

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

### Rodando em segundo plano (pausa até a cota renovar)
- Kaggle `cr-rotulador-9b` v2 (9B rotulador + baselines 9B/2B no teste).
- Local: eficiência do `qwen3.5:9b` (100 avaliações, ~48 min).
- Laço que tenta o push do kernel `cr-treino-qwen08b-lora` a cada 5 min (bloqueado pelo
  limite de 2 sessões de GPU). **O 0.8B só valida o pipeline com os 250 rótulos parciais e
  não entra em nenhum resultado (D28).**
- Alarme às 05:10 de 9/11 para retomar a rotulagem do Gemini.

### Falta
- Rotulagem do Gemini (9/11 de manhã), consolidar, nova versão do dataset do Kaggle.
- Treino 0.8B (pipeline) → treino 2B → GGUF → Ollama local → conferência em 20 exemplos.
- Planilha de revisão (depende do 9B), avaliação final, eficiência do 2B ajustado e do 9B.
- Cards, README final, ESTUDO.md, artigo.

### Para retomar
```bash
conda activate classificador-reclamacoes
python -u src/rotular.py --rotulador gemini --saida outputs/rotulos_gemini.jsonl --tamanho-lote 100 --pausa 15
kaggle kernels status diogomede/cr-rotulador-9b
python src/importar_kaggle.py rotulador_9b          # quando o kernel terminar
kaggle kernels push -p kaggle/treino_qwen08b_lora    # se ainda não tiver sido enviado
```
