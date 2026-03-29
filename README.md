# Global Weather Repository 
## Advanced Data Science Assessment

**PM Accelerator | AI Engineer Internship Technical Assessment**

---

## PM Accelerator Mission

> PM Accelerator empowers aspiring and experienced product managers with the tools, community, and real-world experience needed to accelerate their careers. Through hands-on projects, mentorship, and a global network, PM Accelerator bridges the gap between ambition and achievement.

---

## Project Overview

This project builds a complete machine learning pipeline to **forecast daily temperatures** across 191 countries using the [Global Weather Repository](https://www.kaggle.com/datasets/nelgiriyewithana/global-weather-repository) dataset — 131,365 daily weather snapshots with 41 features.

**Best model:** Gradient Boosting — MAE = 2.61°C, R² = 0.886 on held-out test data (Nov 2025 → Mar 2026)

---

## Repository Structure

```
├── Global_Weather_Repo.ipynb   # Full analysis notebook
├── README.md                   # This file
├── requirements.txt            # Python dependencies
```

> **Note:** The dataset is not included. Download `GlobalWeatherRepository.csv` from [Kaggle](https://www.kaggle.com/datasets/nelgiriyewithana/global-weather-repository) and place it at `/content/GlobalWeatherRepository.csv` (or update the path in Cell 3).

---

## What's Inside the Notebook

### 1. Data Cleaning & Preprocessing
- Parsed `last_updated` datetime and extracted time features
- Removed unit duplicates (°F columns, kph columns, imperial pressure/precipitation)
- Dropped high-correlation redundants confirmed by EDA (r > 0.90)
- Removed 3 physically impossible sensor readings (wind > world record, pressure outside valid range) — 0.002% of data
- Fixed 25+ country name typos discovered during EDA (multilingual API aggregation artifacts)
- **Leakage-safe:** all statistical operations (scaling, imputation) fitted on train set only

### 2. Exploratory Data Analysis
- **Target distribution:** Left-skewed (skewness = -0.88) — explained by geographic concentration of cities in warm latitudes
- **Seasonality:** Clear 12-month cycle; Northern/Southern hemisphere seasons cancel globally, justifying cyclical encoding
- **Correlation analysis:** UV index (r=0.489) and pressure (r=-0.421) are strongest predictors; `year` dropped after discovering a sampling artifact (uneven city coverage across years)
- **Geography:** Latitude vs mean temperature per city — desert/altitude effects explain why latitude alone gives only r=-0.38
- **Anomaly detection:** Three methods compared — Z-score (7%), IQR (37.5%), Isolation Forest (3%); IF used as primary method for multivariate anomaly detection

### 3. Feature Engineering
- **Cyclical time encoding:** sin/cos transforms for month, hour, and day-of-year — preserves adjacency of Dec/Jan, 23:00/00:00
- **Anomaly flag:** `is_anomaly` composite feature — model learns from extreme weather events rather than ignoring them

### 4. Forecasting Models

| Model | MAE (°C) | RMSE (°C) | R² |
|---|---|---|---|
| Baseline (mean) | 9.47 | 12.85 | -0.326 |
| Ridge Regression | 6.01 | 8.00 | 0.487 |
| Random Forest | 3.29 | 4.75 | 0.819 |
| **Gradient Boosting** | **2.61** | **3.76** | **0.886** |
| Stacking Ensemble | 3.00 | 4.21 | 0.858 |

Time-based 80/20 split: train on past (May 2024 → Nov 2025), test on future (Nov 2025 → Mar 2026).

### 5. Advanced Analyses
- **Feature importance:** Random Forest impurity + SHAP values — latitude and day-of-year cyclical encoding are dominant features
- **Climate analysis:** Seasonal patterns by hemisphere, temperature range by climate zone (tropical/subtropical/temperate/polar)
- **Air quality:** PM2.5 vs visibility, ozone vs temperature, pollutant correlations with weather parameters
- **Spatial analysis:** Interactive world map of mean temperature and precipitation per city

---

## How to Run

### Option A — Google Colab (recommended)
1. Upload `Global_Weather_Repo.ipynb` to [colab.research.google.com](https://colab.research.google.com)
2. Upload `GlobalWeatherRepository.csv` to `/content/`
3. Runtime → Run all

### Option B — Local
```bash
git clone <repo-url>
cd <repo-folder>
pip install -r requirements.txt
jupyter notebook Global_Weather_Repo.ipynb
```
Update the CSV path in Cell 3 from `/content/GlobalWeatherRepository.csv` to your local path.

---

## Key Findings

1. **Latitude alone is insufficient** — desert cities at 25°N (Riyadh, Kuwait) are as hot as tropical cities despite higher latitude. Longitude and climate zone context matter.
2. **Month has near-zero global correlation with temperature** — Northern and Southern hemisphere seasons cancel out. Cyclical encoding combined with latitude captures the real signal.
3. **Year was dropped** — appeared to negatively correlate with temperature, but this was a sampling artifact: 2024 had 47 more cities than 2025, artificially inflating 2024 means.
4. **Isolation Forest outperforms univariate methods** — catches unusual combinations (high temperature + high humidity) invisible to Z-score and IQR applied column-by-column.
5. **2.61°C MAE is strong for global forecasting** — the dataset spans a 62°C temperature range across 191 countries; 2.61°C represents ~4.2% of that range.

---

## Requirements

See `requirements.txt` for full list. Core dependencies:
- pandas, numpy, matplotlib, seaborn, plotly
- scikit-learn, shap

---

## Author
Nayra Saadawy

Built as part of the PM Accelerator AI Engineer Internship Technical Assessment.
