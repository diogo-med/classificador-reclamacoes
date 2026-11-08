# Dados

## Origem
[Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce),
licença **CC BY-NC-SA 4.0** (conferida nos metadados do Kaggle em 08/11/2026). Pedidos de 2016 a 2018.

Os CSVs brutos não ficam no git. Para baixar:

```bash
cd data/raw && kaggle datasets download olistbr/brazilian-ecommerce --unzip
```

## Pastas
- `raw/` (fora do git): CSVs originais.
- `interim/` (fora do git): `avaliacoes_com_texto.parquet`, saída de `src/preparar_dados.py`.
- `processed/` (no git): amostra de 4.000 avaliações com split e rótulos. É um subconjunto
  derivado do Olist, redistribuído sob os mesmos termos (CC BY-NC-SA 4.0, com atribuição à Olist).
- `revisao/` (no git): planilha de revisão manual e instruções.
