"""Converte o checkpoint .jsonl de um rotulador num parquet em data/processed/.

Uso:
  python src/consolidar.py outputs/rotulos_gemini.jsonl data/processed/rotulos_gemini.parquet
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rotular import consolidar  # noqa: E402

if __name__ == "__main__":
    origem, destino = Path(sys.argv[1]), Path(sys.argv[2])
    df = consolidar(origem)
    df.to_parquet(destino, index=False)
    print(f"{len(df)} rótulos -> {destino}")
