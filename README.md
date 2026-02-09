# ⚡ Energy Forecasting with XGBoost & CatBoost

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-green.svg)](https://xgboost.readthedocs.io/)
[![CatBoost](https://img.shields.io/badge/CatBoost-1.2-orange.svg)](https://catboost.ai/)
[![Prophet](https://img.shields.io/badge/Prophet-Forecasting-purple.svg)](https://facebook.github.io/prophet/)

**Predicción de consumo energético** usando técnicas de series temporales. Comparativa de 4 modelos: LSTM, Prophet, XGBoost y **CatBoost** (ganador).

## 🎯 Objetivos del Proyecto

- Limpiar y preparar datos horarios de consumo energético
- Detectar estacionalidad y estacionariedad (ADF test, descomposición)
- Comparar rendimiento de **deep learning** vs **gradient boosting** vs **modelos estadísticos**
- Identificar el modelo más preciso y eficiente para forecasting energético

## 📊 Dataset

**Consumo energético horario** (valores reales de kWh):
- **Período:** 2002–2016 
- **Train:** 2002–2015 (~100K observaciones)
- **Test:** 2015–2016 (~12K observaciones)
- **Frecuencia:** Horaria → agregada a diaria para algunos modelos

**Preprocesamiento aplicado:**
- Interpolación de valores faltantes (<1%)
- Detección/eliminación de duplicados
- Creación de **lags** (1,7,30 días) y **rolling windows**

## 🔬 Metodología

EDA → Tendencia ↑, estacionalidad diaria/semanal/anual

Stationarity → ADF test: p-value < 0.05 (estacionaria)

Feature Engineering → Lags, rolling means, dummies festivos

Modelado → 4 algoritmos comparados

Evaluación → MAE, RMSE, MAPE en test set

text

## 📈 Resultados

### **Comparativa de Modelos**

| Modelo | MAE ↓ | RMSE ↓ | MAPE ↓ | Tiempo Entrenamiento |
|--------|-------|--------|--------|---------------------|
| **LSTM** | 58.2 | 74.1 | 12.3% | **120 min** |
| **Prophet** | 52.4 | 68.3 | **9.8%** | 8 min |
| **XGBoost** | **48.7** | **64.4** | 10.2% | 3 min |
| **CatBoost** ⭐ | **44.1** | **58.0** | **9.5%** | **2 min** |

### **🏆 Ganador: CatBoost**
✅ Mejor precisión (RMSE ↓15% vs LSTM)
✅ Más rápido (60x vs LSTM)
✅ Menos tuning requerido
✅ Manejo automático de categóricas

text

**Insights clave:**
- **Gradient boosting > deep learning** para series temporales medianas
- **Lags 7/30 días** fueron las features más predictivas
- **Festivos + temperatura** impactaron +20% en error

## 🛠️ Tech Stack Completo

Core ML: XGBoost 2.0, CatBoost 1.2, scikit-learn
Time Series: Prophet 1.1, statsmodels
Deep Learning: TensorFlow 2.12, Keras
Data: Pandas, NumPy
Viz: Matplotlib, Seaborn, Plotly

text

## 🚀 Instalación y Uso

```bash
git clone https://github.com/jorgegalanr/energy-forecasting-xgboost-catboost.git
cd energy-forecasting-xgboost-catboost
pip install -r requirements.txt
jupyter notebook notebooks/energy_forecasting.ipynb
📁 Estructura del Proyecto
text
├── data/
│   ├── energy_hourly.csv
│   └── energy_daily.csv
├── notebooks/
│   └── energy_forecasting.ipynb
├── src/
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   └── model_comparison.py
├── models/
│   ├── catboost_best.pkl
│   └── xgboost_best.pkl
├── figures/
│   ├── forecast_comparison.png
│   └── feature_importance.png
└── requirements.txt
🎯 Aplicaciones Reales
text
🏭 Utilities: Optimización de generación eléctrica
🏢 Edificios: Gestión inteligente de consumo
🌡️ Smart Grids: Predicción de demanda por zonas
💰 Trading: Arbitraje de precios energéticos

👤 Autor
Jorge Galán Rodríguez
💼 linkedin.com/in/jorgegalanrodriguez
🐱 https://github.com/jorgegalanr
jorgegalanrodriguez@gmail.com
