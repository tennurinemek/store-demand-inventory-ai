"""
Modeli basit referans yöntemlerle karşılaştırır.

Bir tahmin modelinin WAPE'i tek başına anlam taşımaz; "neyi yenmiş?" sorusunun
cevabı gerekir. Bu script aynı 2017 test seti üzerinde üç naif yöntem hesaplar.

Kullanım:
    python src/baselines.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
TEST_START = "2017-01-01"


def wape(y_true, y_pred) -> float:
    return float(np.abs(y_true - y_pred).sum() / y_true.sum() * 100)


def main() -> None:
    df = pd.read_csv(DATA / "train.csv", parse_dates=["date"]).sort_values(
        ["store", "item", "date"]
    )
    grp = df.groupby(["store", "item"])["sales"]
    df["lag_7"] = grp.shift(7)
    df["lag_364"] = grp.shift(364)  # 52 hafta: hem yıllık hem haftalık örüntüyü korur

    test = df[df["date"] >= TEST_START].dropna(subset=["lag_7", "lag_364"]).copy()

    # 2016'nın mağaza-ürün-ay-haftagünü ortalaması
    train_2016 = df[(df["date"] >= "2016-01-01") & (df["date"] < TEST_START)].copy()
    train_2016["month"] = train_2016["date"].dt.month
    train_2016["dow"] = train_2016["date"].dt.dayofweek
    seasonal_mean = (
        train_2016.groupby(["store", "item", "month", "dow"])["sales"]
        .mean()
        .rename("seasonal_mean")
        .reset_index()
    )
    test["month"] = test["date"].dt.month
    test["dow"] = test["date"].dt.dayofweek
    test = test.merge(seasonal_mean, on=["store", "item", "month", "dow"], how="left")

    results = {
        "Naif (7 gün önceki satış)": wape(test["sales"], test["lag_7"]),
        "Mevsimsel naif (364 gün önce)": wape(test["sales"], test["lag_364"]),
        "2016 mağaza-ürün-ay-gün ortalaması": wape(test["sales"], test["seasonal_mean"]),
    }

    forecasts_path = DATA / "forecasts_2017.parquet"
    if forecasts_path.exists():
        fc = pd.read_parquet(forecasts_path)
        results["LightGBM (bu proje)"] = wape(fc["sales"], fc["prediction"])

    print(f"{'Yöntem':<38} {'WAPE':>8}")
    print("-" * 47)
    for name, score in results.items():
        print(f"{name:<38} {score:>7.2f}%")


if __name__ == "__main__":
    main()
