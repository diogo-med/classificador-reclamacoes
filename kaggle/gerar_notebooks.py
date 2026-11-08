"""Converte kaggle/*/notebook.py (formato jupytext percent) em notebook.ipynb,
inserindo o trecho comum de kaggle/comum_ollama.py onde houver <<COMUM_OLLAMA>>.

Uso: python kaggle/gerar_notebooks.py
"""
from pathlib import Path

import jupytext

PASTA = Path(__file__).resolve().parent
COMUM = (PASTA / "comum_ollama.py").read_text()

for fonte in sorted(PASTA.glob("*/notebook.py")):
    texto = fonte.read_text().replace("# <<COMUM_OLLAMA>>", COMUM)
    nb = jupytext.reads(texto, fmt="py:percent")
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    jupytext.write(nb, fonte.with_suffix(".ipynb"))
    print("gerado", fonte.with_suffix(".ipynb"))
