# classificador-reclamacoes

> Em construção. Os números deste projeto ainda não existem; quando existirem, virão de arquivos
> em `resultados/` gerados pelo código.

Fine-tuning com LoRA de um modelo pequeno (Qwen3.5-2B) para classificar avaliações reais de
clientes brasileiros do dataset Olist (e-commerce, 2016-2018) e devolver JSON com sentimento,
categorias de problema e um resumo.

**Pergunta central:** medido em rótulos revisados por um humano, quanto da qualidade de um LLM
grande (Gemini) um modelo de 2B ajustado retém, e a que custo e latência?

**Isto é destilação:** os rótulos de treino são gerados pelo Gemini, e o modelo pequeno aprende a
imitá-lo. Os rótulos do teste também vêm do Gemini e são **provisórios** até a revisão humana.

Andamento em [PROGRESSO.md](PROGRESSO.md); decisões em [DECISOES.md](DECISOES.md); plano em
[docs/plano.md](docs/plano.md).
