# Progresso

Atualizado ao fim de cada etapa. Se a sessão cair, retomar a partir daqui.

## Estado em 09/11/2026, 03:50

Conferido às 03:49: `cr-rotulador-9b` e `cr-treino-qwen08b-lora` com status COMPLETE e saídas
importadas; nenhum kernel rodando. Próximo passo: alarme das 05:10 → `checar_lote.py` → rotulagem.

### Estado em 08/11/2026, 23:40

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
