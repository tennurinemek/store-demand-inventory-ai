#  AI-Powered Demand Forecasting & Dynamic Inventory Optimization

An end-to-end Decision Support System (DSS) integrating **LightGBM regression** with industrial engineering **dynamic inventory policies (Safety Stock & ROP)** to mitigate stockouts and optimize working capital in retail operations.

---

##  Business Problem & Overview

In multi-echelon retail supply chains, static reorder policies fail to adapt to seasonal demand shifts. This project bridges predictive machine learning with prescriptive decision-making:
- **Predictive Phase:** Generates SKU-store level daily demand forecasts.
- **Prescriptive Phase:** Translates forecast variance into dynamic Safety Stock ($SS$) and Reorder Point ($ROP$) thresholds.
- **Actionable Execution:** Interactive Streamlit simulation dashboard triggering automatic replenishment alerts based on daily inventory depletion.

---

##  Core Methodology

### 1. Demand Forecasting (LightGBM Regression)
- **Feature Engineering:** Multi-lag structures ($t-7, t-14, t-21, t-60$), rolling window statistics (7, 14, 30, 60 days), and calendar seasonality.
- **Evaluation Metric:** Optimized for **WAPE (Weighted Absolute Percentage Error)**:
  $$\text{WAPE} = \frac{\sum |y_t - \hat{y}_t|}{\sum y_t} \times 100$$
- **Model Performance:** Baseline WAPE of **~10.41%**.

### 2. Dynamic Inventory Optimization
- **Dynamic Safety Stock ($SS$):**
  $$SS = Z \times \sigma_{\text{error}} \times \sqrt{L}$$
- **Dynamic Reorder Point ($ROP$):**
  $$ROP = (\bar{d}_{\text{forecast}} \times L) + SS$$

---

##  Streamlit Dashboard Features

- **Granular Analysis:** Store and SKU-level forecast vs. actual demand breakdown.
- **Sensitivity Analysis:** Real-time tuning for Lead Time (1–14 days), Service Level (90%, 95%, 99%), and Batch Size (MOQ).
- **Stock Depletion Simulation:** 14-day rolling inventory simulation flagging `Reorder Alert` or ` Stock Sufficient`.

---

##  Tech Stack

- **Languages & Libraries:** Python, Pandas, NumPy, Scikit-learn, LightGBM, Streamlit, Plotly
- **Version Control:** Git, GitHub

---

##  Author
- **Tennur İnemek** — Industrial Engineering Student
- **GitHub:** [@tennurinemek](https://github.com/tennurinemek)
