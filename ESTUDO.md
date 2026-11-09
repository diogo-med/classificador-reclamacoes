# ESTUDO — o porquê de cada decisão

Guia para explicar o projeto numa entrevista. Os números estão em `resultados/` e no README;
aqui fica o raciocínio.

## 1. O problema em uma frase
Um modelo de 2B parâmetros, ajustado com LoRA nos rótulos de um LLM grande (Gemini), consegue
classificar reclamações de clientes em JSON com qualidade próxima à do LLM grande, mas rodando em
CPU, sem custo de API? A métrica final é medida em rótulos revisados por humano.

## 2. Destilação
- **O que é:** o "professor" (Gemini 3.6 Flash) rotula 4.000 avaliações; o "aluno"
  (Qwen3.5-2B) aprende a reproduzir essas saídas. Não há rótulo humano no treino.
- **Consequência 1:** o aluno herda os erros do professor. O teto realista é "igual ao Gemini",
  não "melhor que o Gemini". Por isso a pergunta é *quanto da qualidade ele retém*.
- **Consequência 2:** enquanto o teste usa rótulos do próprio Gemini, a linha do Gemini na tabela
  é 100% por construção. Só a revisão humana do teste transforma o número em medida de qualidade.
- **Por que vale a pena mesmo assim:** latência e custo. Um 2B em CPU custa zero por chamada,
  não depende de rede nem de cota (a cota gratuita do Gemini 3.6 Flash é de 20 requisições/dia), e
  os dados não saem da empresa.

## 3. Dados e amostragem
- **Olist:** reclamações reais de e-commerce brasileiro (2016-2018). Não é delivery de comida, mas
  o vocabulário de atraso, item faltando, produto errado e atendimento se transfere.
- **Deduplicação:** uma avaliação por pedido (a mais recente) e descarte de `review_id` ligado a
  vários pedidos, porque o atraso real precisa estar ligado a um pedido só.
- **Peso nas notas 1-3:** a população tem 64% de notas 4-5 (quase tudo elogio). Sem peso, as
  categorias de problema ficariam com pouquíssimos exemplos.
- **Teste separado antes de ler qualquer texto:** a divisão usa só a nota. As categorias, os
  exemplos do prompt e os hiperparâmetros saem do treino e da validação. O arquivo da amostra está
  versionado no git, o que prova a ordem.

## 4. Categorias
- Lidas 200 avaliações do treino. Mudei `sem_resposta` para `atendimento`, porque as queixas
  incluíam troca, devolução e cancelamento, não só falta de resposta.
- `atraso` × `nao_recebido` é a fronteira mais ambígua ("ainda não recebi"). A regra escrita:
  as duas quando o cliente diz que não recebeu *e* que o prazo passou.
- `elogio` é exclusivo e `outro` cobre o resto, para toda avaliação ter pelo menos uma categoria.
- **Multirrótulo:** uma avaliação pode ter várias categorias, então a métrica é F1 binário por
  categoria (não acurácia de uma classe).

## 5. Rotulagem
- **Lotes + esquema JSON:** várias avaliações por requisição e `response_json_schema` no Gemini,
  com validação item a item no meu código. Item inválido não entra.
- **Checkpoint por lote em `.jsonl`:** se cair, retoma sem repetir requisição (a cota é preciosa).
- **Segundo rotulador (Qwen3.5-9B):** com o mesmo prompt. A concordância entre os dois mede quão
  ambíguas as categorias são. Kappa baixo numa categoria indica definição fraca, não modelo ruim.
  A concordância no teste não é usada para mudar nada (seria vazamento).
- **Checagem independente:** `atraso_real` vem das datas, não de LLM. Se o rótulo `atraso`
  estiver certo, ele deve se concentrar nos pedidos que realmente atrasaram. É uma medida de
  validade que não depende de nenhum modelo.

## 6. Modelo e LoRA
- **Qwen3.5-2B:** pequeno o bastante para CPU, multilíngue (português bom), com suporte no Unsloth.
  O 0.8B serviu para validar o pipeline barato antes de gastar GPU no 2B.
- **LoRA:** treina matrizes de posto baixo (r=16) somadas aos pesos congelados nas projeções de
  atenção e MLP. São ~1% dos parâmetros, cabe numa T4 gratuita e o adaptador pesa dezenas de MB.
- **16 bits, não QLoRA:** a documentação do Unsloth diz que o Qwen3.5 perde mais que o normal com
  quantização em 4 bits no treino.
- **Perda só na resposta:** formato prompt/completion; o modelo não gasta capacidade aprendendo a
  repetir o prompt.
- **Validação para escolher a época:** `load_best_model_at_end` pela perda de validação. O teste
  não participa.

## 7. Chat template
- O mesmo texto exato antes da resposta no treino e na inferência:
  `<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n`.
- O bloco `<think>` vazio é como o Qwen3.5 sinaliza "sem raciocínio". Se o treino não tivesse esse
  bloco e a inferência tivesse (ou o contrário), o modelo veria um prefixo que nunca viu.
- Conferido de três formas: o prefixo é salvo no Kaggle, a demo usa a mesma constante
  (`src/prompts.py`) e as saídas do Ollama local são comparadas com as do Kaggle em 20 exemplos.

## 8. Quantização na inferência
- **q8_0 para a comparação:** o `qwen3.5:2b` do Ollama é q8_0, então o 2B ajustado também é
  avaliado em q8_0. A diferença entre os dois mede o LoRA, não a quantização.
- **Q4_K_M para a demo:** menor e mais rápido em CPU, com perda pequena. É a versão do Space.

## 9. Métricas
- **F1 por categoria e F1 macro:** o macro dá o mesmo peso a categorias raras (`incompleto`,
  `produto_errado`), que somem no micro.
- **F1 micro e acerto exato:** visão agregada e "acertou tudo na avaliação".
- **Sentimento à parte:** acurácia e F1 macro com 3 classes.
- **% JSON válido e % no esquema:** um classificador que quebra o formato não serve em produção.
  Resposta inválida conta como "nenhuma categoria" (erro), não é descartada.
- **Eficiência:** mesma CPU, mesma amostra de 100, uma chamada por vez, aquecimento descartado;
  média e p95 (a cauda importa para SLA). Custo do Gemini com preço oficial, fonte e data.

## 10. Perguntas prováveis numa entrevista

1. **Por que não usar só o Gemini?** Custo, latência, cota e privacidade. Em volume (milhões de
   avaliações), um 2B local custa só o hardware. A tabela de eficiência quantifica isso.
2. **Se os rótulos vêm do Gemini, como você sabe que estão certos?** Não sei sem revisão: por isso
   há (a) revisão humana do teste, (b) o segundo rotulador e (c) a checagem do atraso pelas datas.
3. **O aluno pode superar o professor?** Nos rótulos do professor, quase nunca. Em rótulos
   humanos, às vezes sim, porque o treino suaviza erros aleatórios do professor. Isso só aparece
   depois da revisão.
4. **Por que LoRA e não fine-tuning completo?** Cabe numa GPU gratuita, treina em minutos, o
   adaptador é pequeno e o risco de esquecer o conhecimento geral é menor. Com 3 mil exemplos, o
   ganho de um ajuste completo seria pequeno.
5. **Como evitou vazamento de teste?** Divisão por nota antes de ler textos, versionada no git;
   categorias, exemplos e hiperparâmetros só do treino e da validação.
6. **Por que F1 macro?** Classes desbalanceadas: o micro seria dominado por `elogio`.
7. **O que é kappa de Cohen?** Concordância descontada da concordância esperada por acaso. 0 =
   acaso, 1 = perfeita. Importa quando uma classe é rara (dois rotuladores que sempre dizem "não"
   concordam 95% sem saber nada).
8. **Por que o TF-IDF?** É o piso: se ele empatar com o LLM, o LLM não se justifica. Treina em
   segundos e é interpretável.
9. **Qual o maior risco de colocar em produção?** Mudança de domínio (comida ≠ e-commerce),
   categorias ambíguas e saídas fora do esquema. Mitigação: validação de esquema na saída,
   decodificação restrita (`format` do Ollama) e monitorar a distribuição de categorias.
10. **O que faria com mais tempo?** Revisar mais rótulos de treino, testar o 4B, calibrar limiares
    por categoria, medir com rótulos humanos cegos (sem sugestão) e testar em dados de delivery.

## 11. Pontos fracos (honestos)
- **Rótulos de LLM** e **revisão vendo a sugestão** (viés de ancoragem): a concordância
  Gemini × humano tende a sair superestimada.
- **Cota do Gemini:** 20 requisições/dia forçou lotes de 100 (o plano previa 20-30) e deixou parte
  da amostra sem rótulo no prazo; ver DECISOES.md D19-D21.
- **Baseline do Gemini com o prompt curto não foi rodado** (500 chamadas = 25 dias de cota); a
  linha do Gemini usa os rótulos do prompt de rotulagem.
- **Domínio:** e-commerce 2016-2018, não delivery de comida.
- **Um só seed e uma só divisão:** sem intervalo de confiança por reamostragem do treino.
- **Teste de 500:** categorias raras têm poucas dezenas de positivos; o F1 delas é instável.
