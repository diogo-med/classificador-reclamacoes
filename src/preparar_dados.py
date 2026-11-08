"""Etapa 2: limpeza das avaliações do Olist.

- Deduplica avaliações (uma por pedido, a mais recente; review_id ligado a
  mais de um pedido é descartado, porque não dá para saber a qual pedido o
  atraso se refere).
- Calcula o atraso real pelas datas (só pedidos entregues).
- Mantém só avaliações com texto.
- Salva estatísticas descritivas em resultados/estatisticas_dados.json.

Uso: python src/preparar_dados.py
"""
import json

import pandas as pd

from config import DATA_INTERIM, DATA_RAW, RESULTADOS, garantir_pastas


def deduplicar_avaliacoes(reviews: pd.DataFrame) -> pd.DataFrame:
    """Uma avaliação por pedido (a de review_answer_timestamp mais recente)
    e descarta review_id associado a mais de um pedido."""
    df = reviews.copy()
    df["review_answer_timestamp"] = pd.to_datetime(df["review_answer_timestamp"])
    df = df.sort_values(["order_id", "review_answer_timestamp", "review_id"])
    df = df.drop_duplicates(subset="order_id", keep="last")
    df = df[~df["review_id"].duplicated(keep=False)]
    return df.reset_index(drop=True)


def calcular_atraso(orders: pd.DataFrame) -> pd.DataFrame:
    """Retorna order_id, dias_atraso e atraso_real.

    atraso_real = 1 se a entrega real (data) foi depois da data estimada,
    0 se foi no prazo, e NA se o pedido não foi entregue (status diferente de
    'delivered' ou sem data de entrega)."""
    df = orders[["order_id", "order_status", "order_delivered_customer_date",
                 "order_estimated_delivery_date"]].copy()
    entregue = pd.to_datetime(df["order_delivered_customer_date"]).dt.normalize()
    estimada = pd.to_datetime(df["order_estimated_delivery_date"]).dt.normalize()
    valido = (df["order_status"] == "delivered") & entregue.notna() & estimada.notna()
    df["dias_atraso"] = (entregue - estimada).dt.days.where(valido)
    df["atraso_real"] = (df["dias_atraso"] > 0).astype("Int64").where(valido)
    return df[["order_id", "order_status", "dias_atraso", "atraso_real"]]


def montar_texto(titulo, mensagem) -> str:
    """Junta título e mensagem num texto só. Retorna '' se não houver mensagem."""
    msg = "" if pd.isna(mensagem) else " ".join(str(mensagem).split())
    tit = "" if pd.isna(titulo) else " ".join(str(titulo).split())
    if not msg:
        return ""
    return f"{tit}. {msg}" if tit else msg


def preparar(reviews: pd.DataFrame, orders: pd.DataFrame):
    dedup = deduplicar_avaliacoes(reviews)
    atraso = calcular_atraso(orders)
    df = dedup.merge(atraso, on="order_id", how="left")
    df["texto"] = [montar_texto(t, m) for t, m in
                   zip(df["review_comment_title"], df["review_comment_message"])]
    com_texto = df[df["texto"].str.len() > 0].copy()
    colunas = ["review_id", "order_id", "review_score", "texto", "order_status",
               "dias_atraso", "atraso_real", "review_answer_timestamp"]
    com_texto = com_texto[colunas].rename(columns={"review_score": "nota"})

    entregues = atraso[atraso["atraso_real"].notna()]
    stats = {
        "avaliacoes_brutas": int(len(reviews)),
        "pedidos_brutos": int(len(orders)),
        "avaliacoes_apos_dedup": int(len(dedup)),
        "removidas_na_dedup": int(len(reviews) - len(dedup)),
        "avaliacoes_com_texto": int(len(com_texto)),
        "fracao_com_texto": round(len(com_texto) / len(dedup), 4),
        "pedidos_entregues_com_datas": int(len(entregues)),
        "taxa_atraso_entregues": round(float(entregues["atraso_real"].mean()), 4),
        "com_texto_e_atraso_conhecido": int(com_texto["atraso_real"].notna().sum()),
        "taxa_atraso_com_texto": round(float(com_texto["atraso_real"].dropna().astype(float).mean()), 4),
        "distribuicao_nota_com_texto": {int(k): int(v) for k, v in
                                        com_texto["nota"].value_counts().sort_index().items()},
        "taxa_atraso_por_nota_com_texto": {
            int(k): round(float(v), 4) for k, v in
            com_texto.dropna(subset=["atraso_real"]).groupby("nota")["atraso_real"]
            .apply(lambda s: s.astype(float).mean()).items()},
        "tamanho_texto_caracteres": {
            k: round(float(v), 1) for k, v in
            com_texto["texto"].str.len().describe(percentiles=[.5, .9, .99]).items()},
    }
    return com_texto.reset_index(drop=True), stats


def main():
    garantir_pastas()
    reviews = pd.read_csv(DATA_RAW / "olist_order_reviews_dataset.csv")
    orders = pd.read_csv(DATA_RAW / "olist_orders_dataset.csv")
    com_texto, stats = preparar(reviews, orders)
    com_texto.to_parquet(DATA_INTERIM / "avaliacoes_com_texto.parquet", index=False)
    with open(RESULTADOS / "estatisticas_dados.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
