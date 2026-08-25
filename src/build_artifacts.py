"""
Modelin 2017 test seti tahminlerini ve stok parametrelerini üretir.

Girdi : data/train.csv, models/lgb_demand_model.pkl
Çıktı : data/forecasts_2017.parquet   -> tarih/mağaza/ürün bazında gerçek satış + model tahmini
         data/item_metrics.csv         -> mağaza-ürün bazında gerçek WAPE, MAE, hata std'si

Kullanım:
    python src/build_artifacts.py
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MODELS = ROOT / "models"

LAGS = [7, 14, 21, 30, 60, 90]
WINDOWS = [7, 14, 30, 60]
TEST_START = "2017-01-01"


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Notebook'taki öznitelik üretimiyle birebir aynı adımlar."""
    df = df.sort_values(["store", "item", "date"]).reset_index(drop=True)

    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["is_weekend"] = (df["date"].dt.dayofweek >= 5).astype(int)
    df["day_of_year"] = df["date"].dt.dayofyear
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["quarter"] = df["date"].dt.quarter
    df["day_of_month"] = df["date"].dt.day
    df["day_of_week_num"] = df["date"].dt.dayofweek

    grp = df.groupby(["store", "item"])["sales"]
    for lag in LAGS:
        df[f"sales_lag_{lag}"] = grp.shift(lag)

    # Rolling istatistikler lag_7 üzerinden: model asla t anındaki satışı görmez.
    grp7 = df.groupby(["store", "item"])["sales_lag_7"]
    for w in WINDOWS:
        df[f"sales_roll_mean_{w}"] = grp7.transform(lambda s, w=w: s.rolling(w).mean())
        df[f"sales_roll_std_{w}"] = grp7.transform(lambda s, w=w: s.rolling(w).std())

    return df.dropna().reset_index(drop=True)


def wape(y_true: pd.Series, y_pred: pd.Series) -> float:
    total = y_true.sum()
    return float("nan") if total == 0 else float(np.abs(y_true - y_pred).sum() / total * 100)


def main() -> None:
    raw = pd.read_csv(DATA / "train.csv", parse_dates=["date"])
    df = build_features(raw)

    model = joblib.load(MODELS / "lgb_demand_model.pkl")
    features = list(model.feature_name_)

    test = df[df["date"] >= TEST_START].copy()
    test["prediction"] = model.predict(test[features]).clip(min=0)
    test["error"] = test["sales"] - test["prediction"]

    print(f"Test seti     : {len(test):,} satır ({test.date.min().date()} - {test.date.max().date()})")
    print(f"Genel WAPE    : %{wape(test['sales'], test['prediction']):.2f}")
    print(f"Genel MAE     : {np.abs(test['error']).mean():.2f} adet")

    metrics = (
        test.groupby(["store", "item"])
        .apply(
            lambda g: pd.Series(
                {
                    "error_std": g["error"].std(),
                    "mae": g["error"].abs().mean(),
                    "wape": wape(g["sales"], g["prediction"]),
                    "mean_forecast": g["prediction"].mean(),
                    "mean_actual": g["sales"].mean(),
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )

    DATA.mkdir(exist_ok=True)
    out_cols = ["date", "store", "item", "sales", "prediction"]
    test[out_cols].to_parquet(DATA / "forecasts_2017.parquet", index=False)
    metrics.to_csv(DATA / "item_metrics.csv", index=False)

    print(f"\nÜrün bazlı WAPE aralığı: %{metrics.wape.min():.2f} - %{metrics.wape.max():.2f}")
    print(f"Yazıldı: {DATA / 'forecasts_2017.parquet'}")
    print(f"Yazıldı: {DATA / 'item_metrics.csv'}")


if __name__ == "__main__":
    main()
