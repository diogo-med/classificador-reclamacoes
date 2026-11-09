import pandas as pd

from preparar_dados import calcular_atraso, deduplicar_avaliacoes, montar_texto


def test_dedup_mantem_mais_recente_por_pedido():
    reviews = pd.DataFrame({
        "review_id": ["r1", "r2", "r3"],
        "order_id": ["o1", "o1", "o2"],
        "review_answer_timestamp": ["2018-01-01 10:00", "2018-01-05 10:00", "2018-01-02 10:00"],
    })
    out = deduplicar_avaliacoes(reviews)
    assert sorted(out["review_id"]) == ["r2", "r3"]


def test_dedup_descarta_review_ligada_a_varios_pedidos():
    reviews = pd.DataFrame({
        "review_id": ["r1", "r1", "r2"],
        "order_id": ["o1", "o2", "o3"],
        "review_answer_timestamp": ["2018-01-01", "2018-01-01", "2018-01-01"],
    })
    out = deduplicar_avaliacoes(reviews)
    assert out["review_id"].tolist() == ["r2"]


def test_atraso_compara_datas_sem_hora():
    orders = pd.DataFrame({
        "order_id": ["no_prazo_mesmo_dia", "atrasado", "adiantado", "nao_entregue"],
        "order_status": ["delivered", "delivered", "delivered", "shipped"],
        "order_delivered_customer_date": ["2018-01-10 23:59:00", "2018-01-12 08:00:00",
                                          "2018-01-05 08:00:00", None],
        "order_estimated_delivery_date": ["2018-01-10 00:00:00", "2018-01-10 00:00:00",
                                          "2018-01-10 00:00:00", "2018-01-10 00:00:00"],
    })
    out = calcular_atraso(orders).set_index("order_id")
    assert out.loc["no_prazo_mesmo_dia", "atraso_real"] == 0
    assert out.loc["atrasado", "atraso_real"] == 1
    assert out.loc["atrasado", "dias_atraso"] == 2
    assert out.loc["adiantado", "atraso_real"] == 0
    assert pd.isna(out.loc["nao_entregue", "atraso_real"])


def test_montar_texto():
    assert montar_texto("Ótimo", "chegou  rápido\n") == "Ótimo. chegou rápido"
    assert montar_texto(None, "ok") == "ok"
    assert montar_texto("Só título", None) == ""
