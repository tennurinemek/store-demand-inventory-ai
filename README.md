# Talep Tahmini ve Dinamik Stok Optimizasyonu

LightGBM ile SKU-mağaza bazında günlük talep tahmini yapan ve bu tahminin **hatasından**
dinamik güvenlik stoğu (SS) ile yeniden sipariş noktası (ROP) türeten bir karar destek
sistemi. Streamlit paneli, politikayı gerçek talep verisi üzerinde simüle ederek
gerçekleşen hizmet seviyesini ve stok maliyetini ölçer.

---

## Neden bu proje

Tahmin modelleri genellikle bir hata metriğiyle biter. Oysa operasyonda asıl soru şudur:
*bu hata payıyla kaç adet stok tutmam gerekir?* Bu proje o iki adımı birleştirir —
tahmin hatasının standart sapması, doğrudan güvenlik stoğu formülünün girdisi olur.

## Model performansı

2017 test seti, 182.500 gözlem. Bir WAPE değeri tek başına anlam taşımadığı için
referans yöntemlerle birlikte verilmiştir:

| Yöntem | WAPE |
|---|---:|
| Naif (7 gün önceki satış) | %15,03 |
| Mevsimsel naif (364 gün önce) | %14,54 |
| 2016 mağaza-ürün-ay-gün ortalaması | %11,49 |
| **LightGBM (bu proje)** | **%10,41** |

MAE: 6,12 adet. Ürün bazında WAPE %7,3 ile %22,4 arasında değişiyor.
Tabloyu `python src/baselines.py` ile yeniden üretebilirsiniz.

## Yöntem

**1. Talep tahmini**
- Gecikme (lag) öznitelikleri: t-7, t-14, t-21, t-30, t-60, t-90
- Hareketli ortalama ve standart sapma: 7, 14, 30, 60 gün — tamamı `sales_lag_7` üzerinden
  hesaplanır, böylece model tahmin anındaki satışı hiçbir zaman görmez
- Takvim öznitelikleri: ay, haftanın günü, yılın günü, çeyrek, hafta sonu
- Zaman bazlı bölme: 2013–2016 eğitim, 2017 test

**2. Stok politikası**
- Güvenlik stoğu: `SS = Z × σ_hata × √L`
- Sipariş noktası her gün yeniden hesaplanır:
  `ROP_t = (t+1 … t+L günlerinin model tahminleri toplamı) + SS`

**3. Simülasyon**

Panel bir (s, Q) sürekli gözden geçirme politikasını gerçek 2017 talebi üzerinde işletir:
siparişler `L` gün sonra teslim alınır, yoldaki stok sipariş pozisyonuna dahil edilerek
mükerrer sipariş engellenir, karşılanamayan talep kayıp satış olarak kaydedilir.
Çıktı olarak gerçekleşen hizmet seviyesi, stoksuz gün sayısı, sipariş sıklığı ve
ortalama elde stok raporlanır — yani politikanın hedefini tutturup tutturmadığı ölçülür.

## Kurulum

```bash
git clone https://github.com/tennurinemek/store-demand-inventory-ai.git
cd store-demand-inventory-ai
pip install -r requirements.txt
streamlit run app.py
```

Panel, depoda hazır bulunan `data/forecasts_2017.parquet` dosyasıyla çalışır.

Modeli sıfırdan eğitmek için Kaggle veri setini indirip `data/train.csv` olarak
kaydedin, ardından:

```bash
jupyter notebook notebooks/01_demand_forecasting.ipynb   # keşif ve eğitim
python src/build_artifacts.py                            # tahmin ve metrik dosyaları
```

## Dizin yapısı

```
├── app.py                          Streamlit paneli
├── data/
│   ├── forecasts_2017.parquet      Modelin 2017 tahminleri + gerçekleşen satış
│   └── item_metrics.csv            Mağaza-ürün bazında WAPE, MAE, hata std'si
├── models/lgb_demand_model.pkl     Eğitilmiş LightGBM modeli
├── notebooks/01_demand_forecasting.ipynb
└── src/
    ├── build_artifacts.py          Tahminleri ve stok parametrelerini üretir
    └── baselines.py                Referans yöntem karşılaştırması
```

## Sınırlılıklar

Bu bölüm bilinçli olarak eklenmiştir; sonuçların hangi varsayımlar altında geçerli
olduğunu belirtmeden model çıktısı paylaşmak yanıltıcı olur.

- **Veri sentetiktir.** Kaggle *Store Item Demand Forecasting Challenge* veri seti fiyat,
  promosyon, kampanya veya geçmiş stoksuzluk bilgisi içermez. Gerçek perakende talebi
  bundan belirgin şekilde daha düzensizdir; buradaki WAPE gerçek bir mağazada beklenecek
  değerin altındadır.
- **Tahmin ufku 7 gündür.** Modelin en kısa gecikmesi 7 gün olduğu için 7 günden uzun
  tedarik sürelerinde tahminlerin tekrarlı (recursive) olarak uzatılması gerekir. Panel
  L > 7 seçildiğinde bu uyarıyı gösterir.
- **√L varsayımı.** Formül, günlük tahmin hatalarının birbirinden bağımsız olduğunu
  varsayar. Hatalar otokorelasyonluysa gerçek güvenlik stoğu ihtiyacı daha yüksektir.
- **Tek kademeli envanter.** Depo–mağaza gibi çok kademeli (multi-echelon) yapı
  modellenmemiştir.
- Karşılanamayan talep kayıp satış sayılır; ertelenmiş talep (backorder) senaryosu yoktur.

## Veri kaynağı

[Store Item Demand Forecasting Challenge](https://www.kaggle.com/c/demand-forecasting-kernels-only) — Kaggle.
10 mağaza × 50 ürün, 1 Ocak 2013 – 31 Aralık 2017, 913.000 satır.

## Teknolojiler

Python · pandas · NumPy · scikit-learn · LightGBM · Streamlit · Plotly

## Yazar

**Tennur İnemek** — Endüstri Mühendisliği öğrencisi
[@tennurinemek](https://github.com/tennurinemek)
