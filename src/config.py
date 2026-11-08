"""Caminhos, sementes e constantes compartilhadas por todos os scripts."""
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
DATA_RAW = RAIZ / "data" / "raw"
DATA_INTERIM = RAIZ / "data" / "interim"
DATA_PROCESSED = RAIZ / "data" / "processed"
DATA_REVISAO = RAIZ / "data" / "revisao"
OUTPUTS = RAIZ / "outputs"
RESULTADOS = RAIZ / "resultados"

SEED = 42

# Tamanhos da amostragem (etapa 3)
N_AMOSTRA = 4000
N_TESTE = 500
FRACAO_VALIDACAO = 0.10
# Peso relativo de cada nota na amostragem: notas 1-3 têm mais peso.
PESO_NOTA = {1: 3.0, 2: 3.0, 3: 3.0, 4: 1.0, 5: 1.0}

# Categorias finais (ver DECISOES.md e src/prompts.py para as definições)
CATEGORIAS = [
    "atraso",
    "nao_recebido",
    "incompleto",
    "produto_errado",
    "defeito_qualidade",
    "atendimento",
    "elogio",
    "outro",
]
SENTIMENTOS = ["positivo", "neutro", "negativo"]


def garantir_pastas():
    for p in [DATA_INTERIM, DATA_PROCESSED, DATA_REVISAO, OUTPUTS, RESULTADOS]:
        p.mkdir(parents=True, exist_ok=True)
