# 🚗 Dakar Auto Scraper — Interactive Data App

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B.svg)](https://streamlit.io)
[![Plotly](https://img.shields.io/badge/Plotly-Interactive-3D4E87.svg)](https://plotly.com)
[![Live App](https://img.shields.io/badge/Live%20App-Streamlit-FF4B4B.svg)](METS_TON_LIEN_ICI)

> End-to-end data pipeline that **scrapes**, **cleans**, and **visualizes** car & motorcycle
> listings from dakar-auto.com — the leading vehicle marketplace in Senegal.

## 🌍 What this app does

The Senegalese used-vehicle market is largely informal. This app automates the full pipeline:
scraping → cleaning → storage → interactive visual analytics.

**Use cases**: price benchmarking, mileage-vs-price analysis, brand market share, CSV export
for ML modeling.

## ✨ Features

| Feature | Description |
|---|---|
| 🔍 Live Scraping | Cars, rentals, motorcycles — configurable page count |
| 🧹 Auto Cleaning | Normalizes FCFA prices, km, brands; handles NaN |
| 📊 Dashboard | 10+ interactive Plotly charts |
| 💾 SQLite Storage | Persists data in `dakar_auto_data.db` |
| 📥 CSV Export | One-click download |

## 🚀 Run locally

```bash
git clone https://github.com/Ogoun09gerbad/My_G.git
cd My_G
pip install -r requirements.txt
streamlit run my_data_app.py
```

## 🌐 Deploy on Streamlit Cloud (free, 5 min)

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with GitHub
3. New app → repo: `Ogoun09gerbad/My_G` · branch: `master` · file: `my_data_app.py`
4. Click **Deploy**

> ⚠️ Before deploying: the background image uses a local path. Replace
> `set_bg_image("image/img_file2.jpg")` with a hosted image URL or comment it out.

## 📊 Data schema

`brand` · `model` · `year` · `km` · `fuel` · `gearbox` · `price (FCFA)` · `owner` · `adress` · `category`

## 🛠️ Stack

`streamlit` · `beautifulsoup4` · `requests` · `pandas` · `plotly.express` · `sqlite3`

## 👤 Author

**Géraud Badélé Ogounchi** — Data Scientist · AIMS Senegal
🌐 [Portfolio](https://ogoun09gerbad.github.io) · 💼 [LinkedIn](https://linkedin.com/in/gerogounchi2000)
