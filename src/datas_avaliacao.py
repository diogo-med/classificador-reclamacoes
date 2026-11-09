"""Quantas avaliações foram escritas antes de o pedido ser entregue (checagem de atraso).

Na Olist o cliente pode avaliar antes da entrega. Nesse caso o texto fala de "não recebi" e o
atraso pelas datas só se confirma depois, o que limita quanto o rótulo `atraso` pode cobrir os
pedidos atrasados. Lê os CSVs brutos (data/raw/, fora do git) e grava um resumo pequeno em
resultados/avaliacao_antes_da_entrega.json, que o src/avaliar.py cita na checagem de atraso.

"Antes da entrega" = review_creation_date (dia) anterior ao dia de order_delivered_customer_date.

Uso: python src/datas_avaliacao.py
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATA_PROCESSED, DATA_RAW, RESULTADOS  # noqa: E402


def main():
    amostra = pd.read_parquet(DATA_PROCESSED / "amostra.parquet")
    rev = pd.read_csv(DATA_RAW / "olist_order_reviews_dataset.csv",
                      usecols=["review_id", "order_id", "review_creation_date"])
    ped = pd.read_csv(DATA_RAW / "olist_orders_dataset.csv", usecols=["order_id", "order_delivered_customer_date"])
    m = amostra.merge(rev, on=["review_id", "order_id"], how="left").merge(ped, on="order_id", how="left")
    criada = pd.to_datetime(m["review_creation_date"]).dt.normalize()
    entregue = pd.to_datetime(m["order_delivered_customer_date"]).dt.normalize()
    m["antes_da_entrega"] = criada < entregue
    m["com_data_de_entrega"] = entregue.notna()
    atrasado = m["atraso_real"] == True  # noqa: E712 (coluna anulável)

    def resumo(d):
        return {"n": int(len(d)), "com_data_de_entrega": int(d["com_data_de_entrega"].sum()),
                "avaliadas_antes_da_entrega": int(d["antes_da_entrega"].sum()),
                "atrasados_reais": int(atrasado[d.index].sum()),
                "atrasados_avaliados_antes_da_entrega": int((d["antes_da_entrega"] & atrasado[d.index]).sum())}

    saida = {"definicao": "review_creation_date (dia) < order_delivered_customer_date (dia)",
             "todas": resumo(m), **{s: resumo(g) for s, g in m.groupby("split")}}
    (RESULTADOS / "avaliacao_antes_da_entrega.json").write_text(json.dumps(saida, ensure_ascii=False, indent=2))
    print(json.dumps(saida, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
