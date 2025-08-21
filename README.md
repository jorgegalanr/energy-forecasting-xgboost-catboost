# energy-forecasting-xgboost-catboost

# ⚡ Energy Forecasting with Time Series (XGBoost & CatBoost)

## 📌 Description
This project applies **time series forecasting** techniques to an **energy consumption dataset**.  
Several models were tested (LSTM, Prophet, XGBoost, CatBoost), and performance was evaluated using **MAE** and **RMSE** metrics.

---

## 🎯 Objectives
- Clean and preprocess energy time series data.  
- Check for stationarity and seasonality (ADF test, decomposition).  
- Train and compare multiple models:  
  - LSTM (deep learning)  
  - Prophet (statistical forecasting)  
  - XGBoost, CatBoost (gradient boosting methods)  
- Select the most accurate model for forecasting.  

---

## 📊 Dataset
Energy consumption dataset (hourly values).  
- Preprocessing included handling missing values, duplicates, and resampling.  
- Train: 2002–2015, Test: 2015–end.  

---

## ⚙️ Tech Stack
- Python  
- Pandas, NumPy  
- Scikit-learn  
- TensorFlow/Keras (LSTM)  
- Prophet  
- XGBoost, CatBoost  
- Matplotlib, Seaborn  

---

## 🔎 Methodology
1. **Data Preparation** → checked nulls, duplicates, interpolated missing values.  
2. **Stationarity Test** → ADF confirmed stationary series.  
3. **Decomposition** → observed trend, seasonality, residuals.  
4. **Modeling**:  
   - LSTM (initially high RMSE, improved with MinMax scaling and lags).  
   - Prophet (captured yearly seasonality well).  
   - XGBoost & CatBoost (best performance).  
5. **Evaluation**:  
   - XGBoost: MAE ~48.7, RMSE ~64.4  
   - CatBoost: MAE ~44.1, RMSE ~58.0 (best)  

---

## 📈 Results
- **CatBoost** achieved the best trade-off between accuracy and computational efficiency.  
- **Prophet** worked well for seasonality but less accurate than boosting models.  
- **LSTM** required heavy tuning and was less efficient.

## Resultados versión Español
- El mejor resultado se obtuvo con **CatBoost (MAE ~44, RMSE ~58).
- Prophet capturó bien la estacionalidad anual, pero fue menos preciso.
- LSTM funcionó peor en términos de error y coste computacional. 


Example forecast visualization:  
![Energy Forecast](reports/figures/forecast.png)

---

## 🚀 How to Run
```bash
git clone https://github.com/jorgegalanr/energy-forecasting-xgboost-catboost.git
cd energy-forecasting-xgboost-catboost
pip install -r requirements.txt
jupyter notebook notebooks/energy_forecasting.ipynb
