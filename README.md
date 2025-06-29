# USA Daily Energy Consumption Forecasting with Prophet & NeuralProphet - *American Electric Power (AEP)*

## Project Overview

This project focuses on **forecasting daily energy demand** for the **AEP region**, leveraging advanced time series modeling techniques using the PJM Interconnection dataset. PJM Interconnection LLC is a regional transmission organization (RTO) responsible for coordinating wholesale electricity movement across multiple U.S. states. The dataset provides **hourly energy consumption data**, which is **aggregated to daily levels** to extract broader consumption patterns.

The primary objective is to **predict future energy usage** to support energy grid planning, load balancing, and long-term infrastructure management. This work builds a **robust, interpretable, and scalable pipeline** utilizing:

- **Prophet (with external regressors, hyperparameter tuning, and holidays)**
- **NeuralProphet (with autoregressive memory and recursive forecasting)**
- **Multi-horizon forecasting (7 days, 3 months, 1 year)**

---

**Author**: [Einstein Ebereonwu](https://www.linkedin.com/in/einstein-ebereonwu/) • [GitHub](https://github.com/munas-git)  
**Dataset**: [Hourly Energy Consumption](https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption?select=AEP_hourly.csv)  
**Notebooks**: [GitHub Notebook](https://github.com/munas-git/US-Energy-Consumption-Forecasting-Prophet-NeuralProphet/blob/main/analysis.ipynb) | [Kaggle Version](https://www.kaggle.com/code/munaee/energy-consumption-forecast-prophet-neural-pro)

---

## Pipeline Highlights

| Stage                        | Key Highlights                                                                                                                                                            |
|-----------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Data Preparation**        | • Aggregated hourly -> daily<br>• Checked for missing days<br>• Verified completeness across years<br>• Mapped US holidays using `holidays` lib                             |
| **Feature Engineering**     | • Seasonal OHE<br>• Lag-based AR features <br>• Holiday indicator columns<br>• Multi-season encodings<br>• Recursive forecasting frame setup                     |
| **Stationarity Testing**    | • ADF test + visual checks<br>• Differenced rolling plots<br>• Structural shift inspection                                                                                 |
| **Visual Decomposition**    | • Seasonally grouped boxplots<br>• Trend overlays<br>• ACF/PACF inspection post-modeling                                                                                   |
| **Modeling Techniques**     | • Base Prophet<br>• Hyperparameter-tuned Prophet<br>• NeuralProphet + AR lags<br>• Recursive daily forecast loop<br>• Seasonal regressor injection                         |
| **Diagnostics & Evaluation**| • MAE, RMSE, MAPE, SMAPE<br>• Visual residual inspection<br>• ACF/PACF residuals<br>• Changepoint tuning<br>• Full horizon tracking for 7, 90, 365-day targets             |

---

## Optimization Techniques Used

- **Time-Aware Cross Validation**: Prophet’s `cross_validation()` used with `initial`, `horizon`, and `period` settings to simulate real forecasting scenarios.
- **Hyperparameter Grid Search**: 60+ combinations explored across:
  - `changepoint_prior_scale`
  - `seasonality_prior_scale`
  - `seasonality_mode`
  - `holidays_prior_scale`
- **Holiday Effects Modeling**: Official U.S. holidays injected as events; model learns temporal context of holiday-driven consumption behavior.
- **External Regressors**: Seasonal OHE (`season_spring`, `season_summer`, etc.) integrated into Prophet for contextual modeling.
- **Recursive Forecasting**: NeuralProphet AR model used recursively to build step-by-step daily forecasts into the future.
- **Multi-Horizon Strategy**:
  - **Short-Term (7 days)** - high accuracy focus
  - **Medium-Term (90 days)** - directional insight
  - **Long-Term (1 year)** - trend guidance only

---

## Tech Stack

Python • pandas • Prophet • NeuralProphet • scikit-learn • statsmodels • seaborn • ACF/PACF • holidays • matplotlib • plotly • Git/GitHub

---

## Forecasting Models

### • **Baseline Prophet**
- Time-only trends, no regressors
- Captured general direction but missed seasonal/holiday dips
- Residuals had strong autocorrelation (especially lag-2)

### • **Optimized Prophet with Regressors & Holidays**
- Seasonal one-hot encoding + holidays as regressors
- Changepoint, seasonality, and holiday priors tuned
- Used **time-aware CV for hyper-parameter optimization**
- Substantial reduction in test RMSE and autocorrelated residuals

### • **NeuralProphet Base**
- Built-in seasonality/trend handling
- Slightly outperformed base Prophet
- Lacked long-memory structure

### • **NeuralProphet with Lag Terms**
- Introduced AR(2) lags
- Recursive forecasting strategy
- Achieved **lowest test error metrics**, indicating robust memory modeling

---

## Multi-Horizon Forecasting Results

| Horizon         | Purpose                                  | Accuracy Focus     |
|----------------|-------------------------------------------|--------------------|
| **Short-term**  | Next 7 days - operational accuracy       | Critical focus   |
| **Medium-term** | Next 3 months - strategic planning       | Approximate trend  |
| **Long-term**   | Next 1 year - infrastructure forecasting | Directional only   |

---

## Closing Note

This updated pipeline demonstrates how combining **interpretable models**, **external calendar data**, and **robust CV-tuned hyperparameters** can produce **high-accuracy short-term forecasts**, while also capturing seasonal and holiday-driven trends in long-term energy usage.

It reflects a rigorous modeling strategy that scales beyond the academic realm and is deployable in real-world energy planning and sustainability workflows.
