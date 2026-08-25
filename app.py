"""
AI destekli talep tahmini ve dinamik stok yönetimi paneli.

Gösterilen bütün tahminler LightGBM modelinin 2017 test seti çıktısıdır
(src/build_artifacts.py tarafından üretilir). Panelde hiçbir sayı gerçek
satıştan türetilmez veya elle ölçeklenmez.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

DATA = Path(__file__).parent / "data"

Z_SCORES = {"%90 (Z = 1.28)": 1.28, "%95 (Z = 1.65)": 1.65, "%99 (Z = 2.33)": 2.33}
MODEL_HORIZON = 7  # Modelin en kısa gecikmesi 7 gün: geçerli tahmin ufku 7 gün.

st.set_page_config(
    page_title="Talep Tahmini ve Stok Optimizasyonu",
    page_icon="📦",
    layout="wide",
)


# --------------------------------------------------------------------------- #
# Veri
# --------------------------------------------------------------------------- #
@st.cache_data
def load_assets():
    forecasts = pd.read_parquet(DATA / "forecasts_2017.parquet")
    forecasts["date"] = pd.to_datetime(forecasts["date"])
    metrics = pd.read_csv(DATA / "item_metrics.csv")
    return forecasts, metrics


try:
    forecasts, metrics = load_assets()
except FileNotFoundError:
    st.error(
        "Tahmin dosyaları bulunamadı. Önce `python src/build_artifacts.py` komutunu "
        "çalıştırarak `data/forecasts_2017.parquet` ve `data/item_metrics.csv` "
        "dosyalarını üretin."
    )
    st.stop()


# --------------------------------------------------------------------------- #
# Envanter simülasyonu
# --------------------------------------------------------------------------- #
def simulate_inventory(demand, forecast, lead_time, safety_stock, order_qty):
    """
    (s, Q) sürekli gözden geçirme politikası.

    Her gün:
      1. O gün teslim edilmesi planlanan siparişler depoya girer.
      2. Talep karşılanır; stok yetmezse aradaki fark kayıp satıştır.
      3. Stok pozisyonu (eldeki + yoldaki) sipariş noktasının altındaysa Q adetlik
         sipariş verilir ve `lead_time` gün sonra teslim alınır.

    Yoldaki siparişlerin stok pozisyonuna dahil edilmesi, teslimat beklenirken aynı
    ihtiyaç için tekrar tekrar sipariş açılmasını (over-ordering) önler.

    Sipariş noktası her gün yeniden hesaplanır:
        ROP_t = (t+1 ... t+L günleri için model tahminlerinin toplamı) + SS
    """
    n = len(demand)
    pipeline = np.zeros(n + lead_time + 1)  # pipeline[i]: i. günün başında gelen miktar

    # Başlangıç stoğu: siparişi yeni teslim alınmış bir döngünün tepesi.
    # Düşük bir başlangıç değeri, ilk sipariş L gün sonra geleceği için ölçümü
    # yapay olarak bozar; bu yüzden ilk L gün ayrıca ısınma sayılır.
    initial_lt_demand = forecast[:lead_time].sum() if n >= lead_time else forecast.sum()
    on_hand = float(np.ceil(initial_lt_demand + safety_stock) + order_qty)

    rows = []
    for t in range(n):
        arrived = pipeline[t]
        on_hand += arrived

        sold = min(on_hand, demand[t])
        lost = demand[t] - sold
        on_hand -= sold

        # Tedarik süresi boyunca beklenen talep, modelin kendi tahminlerinden.
        window = forecast[t + 1 : t + 1 + lead_time]
        expected = window.sum() if len(window) else forecast[t] * lead_time
        rop = int(np.ceil(expected + safety_stock))

        position = on_hand + pipeline[t + 1 :].sum()

        ordered = 0
        while position <= rop and ordered < order_qty * 10:  # güvenlik sınırı
            pipeline[t + lead_time] += order_qty
            position += order_qty
            ordered += order_qty

        rows.append(
            {
                "Gelen sipariş": arrived,
                "Talep": demand[t],
                "Karşılanan": sold,
                "Kayıp satış": lost,
                "Gün sonu stok": on_hand,
                "Yoldaki stok": pipeline[t + 1 :].sum(),
                "Sipariş noktası (ROP)": rop,
                "Verilen sipariş": ordered,
            }
        )

    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Kontrol paneli
# --------------------------------------------------------------------------- #
st.sidebar.header("Seçim")
store_id = st.sidebar.selectbox("Mağaza", sorted(forecasts["store"].unique()))
item_id = st.sidebar.selectbox("Ürün", sorted(forecasts["item"].unique()))

st.sidebar.markdown("---")
st.sidebar.header("Tedarik zinciri parametreleri")
lead_time = st.sidebar.slider("Tedarik süresi (gün)", 1, 14, 3)
service_level = st.sidebar.selectbox("Hizmet seviyesi", list(Z_SCORES), index=1)
order_qty = st.sidebar.number_input(
    "Sipariş parti büyüklüğü (MOQ)", min_value=20, max_value=400, value=60, step=10
)
sim_days = st.sidebar.slider("Simülasyon uzunluğu (gün)", 30, 365, 90, step=30)

z_score = Z_SCORES[service_level]
target_service = float(service_level[1:3])

if lead_time > MODEL_HORIZON:
    st.sidebar.warning(
        f"Model {MODEL_HORIZON} gün ileriye tahmin üretiyor. Daha uzun tedarik "
        "süreleri için tahminlerin tekrarlı (recursive) olarak uzatılması gerekir; "
        "bu ayarda sonuçlar iyimser."
    )

# --------------------------------------------------------------------------- #
# Seçilen mağaza-ürün
# --------------------------------------------------------------------------- #
subset = (
    forecasts[(forecasts["store"] == store_id) & (forecasts["item"] == item_id)]
    .sort_values("date")
    .reset_index(drop=True)
)
row = metrics[(metrics["store"] == store_id) & (metrics["item"] == item_id)].iloc[0]

sigma_error = row["error_std"]
item_wape = row["wape"]
item_mae = row["mae"]
mean_forecast = row["mean_forecast"]

# SS = Z × σ_hata × √L
safety_stock = int(np.ceil(z_score * sigma_error * np.sqrt(lead_time)))
static_rop = int(np.ceil(mean_forecast * lead_time + safety_stock))

st.title("Talep tahmini ve dinamik stok yönetimi")
st.caption(
    "LightGBM talep tahminini güvenlik stoğu ve yeniden sipariş noktası politikasıyla "
    "birleştiren karar destek paneli. Tüm tahminler modelin 2017 test seti çıktısıdır."
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Ortalama günlük tahmin", f"{mean_forecast:.1f} adet")
c2.metric("Güvenlik stoğu (SS)", f"{safety_stock} adet")
c3.metric(
    "Sipariş noktası (ROP)",
    f"{static_rop} adet",
    help="Ortalama tahmin üzerinden hesaplanan referans değer. "
    "Simülasyonda her gün yeniden hesaplanır.",
)
c4.metric(
    "Bu ürünün WAPE'i",
    f"%{item_wape:.2f}",
    help=f"MAE: {item_mae:.2f} adet — modelin bu mağaza-ürün için 2017'deki gerçek hatası.",
)

st.markdown("---")

# --------------------------------------------------------------------------- #
# Tahmin - gerçekleşen karşılaştırması
# --------------------------------------------------------------------------- #
st.subheader(f"Mağaza {store_id} · Ürün {item_id} — tahmin ve gerçekleşen (2017)")

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=subset["date"],
        y=subset["sales"],
        name="Gerçekleşen satış",
        mode="lines",
        line=dict(color="#2b5c8f", width=1.4),
    )
)
fig.add_trace(
    go.Scatter(
        x=subset["date"],
        y=subset["prediction"],
        name="Model tahmini",
        mode="lines",
        line=dict(color="#e08214", width=1.4),
    )
)
fig.add_hline(
    y=safety_stock,
    line=dict(color="#d9534f", dash="dash", width=2),
    annotation_text="Güvenlik stoğu",
    annotation_position="top left",
)
fig.update_layout(
    xaxis_title="Tarih",
    yaxis_title="Adet",
    template="plotly_white",
    hovermode="x unified",
    height=380,
    legend=dict(orientation="h", y=1.12),
)
st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------------------------------- #
# Simülasyon
# --------------------------------------------------------------------------- #
st.subheader(f"Envanter simülasyonu — son {sim_days} gün")
st.caption(
    "Stok her gün gerçekleşen talep kadar erir. Sipariş noktasına inildiğinde sipariş "
    f"açılır ve {lead_time} gün sonra teslim alınır."
)

sim_input = subset.tail(sim_days).reset_index(drop=True)
sim = simulate_inventory(
    demand=sim_input["sales"].to_numpy(dtype=float),
    forecast=sim_input["prediction"].to_numpy(dtype=float),
    lead_time=lead_time,
    safety_stock=safety_stock,
    order_qty=int(order_qty),
)
sim.insert(0, "Tarih", sim_input["date"].dt.date)

# İlk L gün ısınma dönemi: başlangıç stoğunun etkisi ölçümden çıkarılır.
measured = sim.iloc[lead_time:]
total_demand = measured["Talep"].sum()
fill_rate = (
    (1 - measured["Kayıp satış"].sum() / total_demand) * 100 if total_demand else 100.0
)
stockout_days = int((measured["Kayıp satış"] > 0).sum())
n_orders = int((measured["Verilen sipariş"] > 0).sum())

m1, m2, m3, m4 = st.columns(4)
m1.metric(
    "Gerçekleşen hizmet seviyesi",
    f"%{fill_rate:.2f}",
    delta=f"{fill_rate - target_service:+.2f} puan (hedefe göre)",
)
m2.metric("Stoksuz gün", f"{stockout_days} gün")
m3.metric("Verilen sipariş", f"{n_orders} kez")
m4.metric("Ortalama elde stok", f"{measured['Gün sonu stok'].mean():.0f} adet")

fig2 = go.Figure()
fig2.add_trace(
    go.Scatter(
        x=sim["Tarih"],
        y=sim["Gün sonu stok"],
        name="Eldeki stok",
        mode="lines",
        fill="tozeroy",
        line=dict(color="#2b5c8f", width=1.6),
    )
)
fig2.add_trace(
    go.Scatter(
        x=sim["Tarih"],
        y=sim["Sipariş noktası (ROP)"],
        name="Sipariş noktası",
        mode="lines",
        line=dict(color="#e08214", dash="dot", width=1.6),
    )
)
fig2.add_hline(
    y=safety_stock,
    line=dict(color="#d9534f", dash="dash", width=1.6),
    annotation_text="Güvenlik stoğu",
)
order_days = sim[sim["Verilen sipariş"] > 0]
fig2.add_trace(
    go.Scatter(
        x=order_days["Tarih"],
        y=order_days["Gün sonu stok"],
        name="Sipariş verildi",
        mode="markers",
        marker=dict(color="#1a7f37", size=8, symbol="triangle-up"),
    )
)
fig2.update_layout(
    xaxis_title="Tarih",
    yaxis_title="Adet",
    template="plotly_white",
    hovermode="x unified",
    height=380,
    legend=dict(orientation="h", y=1.12),
)
st.plotly_chart(fig2, use_container_width=True)

with st.expander("Günlük simülasyon tablosu"):
    st.dataframe(sim.round(1), use_container_width=True, hide_index=True)

with st.expander("Yöntem ve sınırlılıklar"):
    st.markdown(
        f"""
**Formüller**

- Güvenlik stoğu: `SS = Z × σ_hata × √L`. Buradaki `σ_hata`, modelin bu mağaza-ürün
  için 2017'deki günlük tahmin hatalarının standart sapmasıdır ({sigma_error:.2f} adet).
- Sipariş noktası her gün yeniden hesaplanır:
  `ROP_t = (t+1 … t+L günlerinin model tahminleri toplamı) + SS`

**Sınırlılıklar**

- Veri seti Kaggle *Store Item Demand Forecasting Challenge* verisidir ve sentetiktir.
  Fiyat, promosyon, kampanya ve geçmiş stoksuzluk bilgisi içermez; gerçek perakende
  talebi bu veriden daha düzensizdir.
- Modelin en kısa gecikmesi 7 gün olduğundan geçerli tahmin ufku 7 gündür. Daha uzun
  tedarik sürelerinde tahminlerin tekrarlı olarak uzatılması gerekir.
- `√L` çarpanı günlük tahmin hatalarının bağımsız olduğunu varsayar. Hatalar
  otokorelasyonluysa gerçek güvenlik stoğu ihtiyacı bundan yüksektir.
- Simülasyon karşılanamayan talebi kayıp satış (lost sales) sayar; ertelenmiş talep
  (backorder) senaryosu modellenmemiştir.
"""
    )
