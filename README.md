# Energy Forecasting with XGBoost & CatBoost

Proyecto reproducible de forecasting horario que compara XGBoost y CatBoost con referencias ingenuas. La tarea es predecir el consumo de la próxima hora utilizando exclusivamente datos conocidos hasta la hora anterior.

Este repositorio demuestra preparación de series temporales, prevención de fuga de información, validación cronológica y comunicación de métricas. El caso no utiliza información financiera, pero las mismas decisiones metodológicas son aplicables a forecasting de demanda, cobros o tesorería.

## Pregunta analítica

> ¿Cuánto consumo se observará en la siguiente hora, una vez disponible el dato de la hora actual?

El alcance es deliberadamente concreto: **forecast rolling de una hora por delante**. No se presentan las predicciones de varios años como válidas porque, sin observaciones futuras ni variables exógenas, ese escenario exigiría una estrategia recursiva diferente y acumularía error.

## Datos

El archivo `data/energy_train.csv` contiene 124.870 registros horarios entre abril de 2002 y junio de 2016. El proceso reproducible:

- ordena cronológicamente;
- conserva una observación en cada marca temporal duplicada;
- reconstruye una rejilla horaria completa;
- interpola 28 huecos internos;
- valida que no queden valores ausentes ni consumos no positivos.

La fuente original y la unidad física no están documentadas en el material de partida. Por eso el proyecto habla de **unidades de consumo** y declara esta carencia como una limitación, en lugar de atribuir al dato una procedencia o unidad no verificadas.

## Prevención de fuga de información

Las variables predictoras son:

- retardos de 1, 2, 24, 48 y 168 horas;
- media y desviación móvil de 24 y 168 horas;
- ciclos horarios, semanales y anuales codificados con seno y coseno.

Las estadísticas móviles se calculan sobre `Energy.shift(1)`. Así, el valor real de la hora que se intenta estimar no interviene en sus propias variables. La división también es estrictamente temporal:

| Conjunto | Periodo |
|---|---|
| Entrenamiento | abril de 2002 — diciembre de 2015 |
| Test final | enero de 2016 — junio de 2016 |

## Resultados reproducidos

Resultados obtenidos ejecutando `python run_analysis.py`:

| Modelo | MAE | RMSE | WAPE |
|---|---:|---:|---:|
| XGBoost | 50,21 | 65,77 | 0,91 % |
| CatBoost | 50,13 | 65,83 | 0,91 % |
| Persistencia (1 hora) | 153,79 | 199,06 | 2,78 % |
| Referencia semanal (168 horas) | 602,93 | 827,00 | 10,91 % |

XGBoost obtiene el menor RMSE y CatBoost el menor MAE, pero la diferencia entre ambos es pequeña. La conclusión defendible es que ambos superan ampliamente las referencias en este test; no que uno sea universalmente mejor.

![Forecast de la última semana](reports/generated/forecast_last_week.png)

![Importancia de variables](reports/generated/xgboost_feature_importance.png)

## Estructura

```text
.
├── data/energy_train.csv              # Serie histórica incluida
├── notebooks/energy_forecasting.ipynb # Recorrido analítico reproducible
├── reports/generated/                 # Métricas, predicciones y figuras
├── src/
│   ├── data.py                        # Calidad y regularización horaria
│   ├── features.py                    # Variables estrictamente causales
│   ├── evaluation.py                  # Split, referencias y métricas
│   └── modeling.py                    # Configuración de modelos
├── tests/                              # Controles de datos y leakage
└── run_analysis.py                     # Ejecución completa por CLI
```

## Instalación y ejecución

Probado con Python 3.11. Python 3.12 también es compatible con las dependencias declaradas.

```bash
python -m venv .venv
source .venv/bin/activate           # Linux/macOS
# .\.venv\Scripts\Activate.ps1    # Windows PowerShell

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python run_analysis.py
```

Para abrir el análisis narrativo:

```bash
python -m jupyter notebook notebooks/energy_forecasting.ipynb
```

## Pruebas

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

Las pruebas comprueban la limpieza de duplicados y huecos, la división cronológica, las métricas y que las medias móviles excluyen el objetivo actual. GitHub Actions las ejecuta en cada Pull Request.

## Limitaciones

- La procedencia y unidad de la serie original no están verificadas.
- La evaluación es de una hora por delante con actualización de observaciones reales.
- No se utilizan variables exógenas como temperatura, festivos o actividad económica.
- Un despliegue real requeriría monitorización del error, detección de deriva y una política de reentrenamiento.
- Los resultados no deben extrapolarse a forecasting multi-step sin una evaluación específica.

## Autor

Jorge Galán Rodríguez — [GitHub](https://github.com/jorgegalanr)

## Licencia

MIT.
