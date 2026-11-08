# Trecho comum aos notebooks que usam Ollama no Kaggle (copiado para cada notebook
# por kaggle/gerar_notebooks.py). Instala o Ollama, sobe o servidor e baixa os modelos.
import glob, os, subprocess, sys, time, json
from pathlib import Path
import requests

SRC = os.path.dirname(glob.glob("/kaggle/input/**/amostra.parquet", recursive=True)[0])
sys.path.insert(0, SRC)
os.environ["CR_DATA_PROCESSED"] = SRC
SAIDA = Path("/kaggle/working")
print("dados e código em", SRC, sorted(os.listdir(SRC)))
subprocess.run("nvidia-smi", shell=True)


def subir_ollama(modelos):
    subprocess.run("apt-get install -y -qq zstd > /dev/null 2>&1; "
                   "curl -fsSL https://ollama.com/install.sh | sh", shell=True, check=True)
    log = open(SAIDA / "ollama_serve.log", "w")
    subprocess.Popen(["ollama", "serve"], stdout=log, stderr=log)
    for _ in range(60):
        try:
            requests.get("http://localhost:11434/api/version", timeout=2)
            break
        except requests.RequestException:
            time.sleep(2)
    for m in modelos:
        subprocess.run(["ollama", "pull", m], check=True, stdout=subprocess.DEVNULL)
    lista = subprocess.run(["ollama", "list"], capture_output=True, text=True).stdout
    versao = requests.get("http://localhost:11434/api/version").json()
    (SAIDA / "ollama_modelos.txt").write_text(f"{versao}\n{lista}")
    print(versao, lista)
