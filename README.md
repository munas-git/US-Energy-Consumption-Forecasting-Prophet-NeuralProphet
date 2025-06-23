# USA Daily Energy Consumption Forecasting with Prophet & NeuralProphet - *American Electric Power (AEP)*

## Project Overview

This project aims to forecast **daily energy consumption** for the **AEP region** using advanced time series modeling techniques, based on the PJM Interconnection dataset. PJM Interconnection LLC is a regional transmission organization (RTO) managing wholesale electricity flow across multiple states in the Eastern Interconnection grid. The dataset contains historical hourly energy consumption data (in megawatts), which I aggregate to a **daily level** to capture broader consumption trends rather than fine-grained hourly fluctuations.

The primary goal is to **predict future energy usage patterns** to aid energy planning, load balancing, and infrastructure optimization.

Moving beyond traditional time series models, this project leverages the complementary strengths of **Prophet and NeuralProphet**, alongside **ensemble methods such as model stacking**. By systematically comparing their performance and diagnostic metrics, I develop a **robust, interpretable, and scalable** daily forecasting model.

Unlike many black-box approaches, this analysis integrates **classical decomposition**, **residual diagnostics**, and **ACF/PACF-guided refinements**, combined with **Prophet-based signal modeling**, while benchmarking results against **NeuralProphet** and **stacked ensemble regressors**.

**Author**: [Einstein Ebereonwu](https://www.linkedin.com/in/einstein-ebereonwu/) • [GitHub](https://github.com/munas-git)  
**Dataset**: [Hourly Energy Consumption](https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption?select=AEP_hourly.csv)

---

## Pipeline Highlights

| Stage                        | Key Highlights                                                                                                                                                            |
|-----------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Data Preparation**        | • Aggregated hourly -> daily<br>• Extracted year/month/season<br>• Mapped US federal holidays via `holidays` lib                                                            |
| **Feature Engineering**     | • Seasonal one-hot encodings<br>• Holiday indicator injection<br>• Lag-based AR terms (1–10 days)<br>• Calendar-decomposed regressors                                      |
| **Stationarity Testing**    | • Used **ADF Test** to check raw stability<br>• Applied **first-order differencing**<br>• Evaluated seasonal stability visually & statistically                             |
| **Visual Decomposition**    | • Boxplots & scatterplots by season/holiday<br>• Rolling means and ACF/PACF diagnostics<br>• Yearly, monthly trend overlays                                                 |
| **Modeling Techniques**     | • Baseline Prophet (univariate)<br>• Optimized Prophet (multivariate, holiday-aware)<br>• NeuralProphet (with/without AR lags)<br>• Ridge, Lasso, RF & GBoost stack trials |
| **Diagnostics & Evaluation**| • Metrics: MAE, RMSE, MAPE, SMAPE<br>• Residual plots + ACF/PACF to detect signal leakage<br>• Changepoint & uncertainty calibration                                 |

---

## Tech Stack

Python • pandas • Prophet • NeuralProphet • scikit-learn • statsmodels • seaborn • ACF/PACF • holidays • matplotlib

---

## Forecasting Models

The following model versions were built and evaluated with time-based splits and consistent error metrics (MAE, RMSE, MAPE, SMAPE):

- **Baseline Prophet**  
  Used time-only trend with no external regressors. Captured overall trend but missed certain seasonal dips and holiday effects. Residual errors showed consistent autocorrelation.

- **Optimised Prophet with Season / Holiday Effects**  
  Included engineered season indicators and official US holidays. Added prior scale tuning for changepoints and seasonality. Reduced test set errors and residual autocorrelation but not fully eliminated. 

- **NeuralProphet Base**  
  Used default seasonality and trend components. Outperformed base Prophet slightly without additional feature engineering but struggled with structural memory.

- **NeuralProphet with Lag Terms**  
  Integrated 10-day autoregressive memory. Significantly improved accuracy and error independence. Residual diagnostics showed white noise characteristics and minimal lag structure.

---

## Residual Diagnostics

| Model                        | Residual Shape              | ACF/PACF Diagnostic                              |
|-----------------------------|-----------------------------|--------------------------------------------------|
| Prophet (Base)              | Normal, but seasonal spikes | Strong autocorrelation @ multiple lags           |
| Prophet (Optimised)         | Mostly Gaussian             | Lag-1 autocorrelation still present              |
| NeuralProphet (Optimised)   | Nearly white-noise          | ✅ Lags beyond 0 ≈ 0 (residuals are memoryless)   |

---

## Closing Note

This project demonstrates the development of a forecasting approach,from simple time-based models to advanced, signal-aware autoregressive neural methods, within a reproducible and interpretable workflow. When planning for energy grid efficiency and sustainability, the goal is to **predict with a high level of accuracy** in order to **effectively plan and serve** the energy grid’s needs. I have been able to do just that within this analysis.

---

**View full notebook**: [GitHub](https://github.com/munas-git/US-Energy-Consumption-Forecasting-Prophet-NeuralProphet/blob/main/analysis.ipynb) | [Kaggle Notebook](https://www.kaggle.com/code/munaee/energy-consumption-forecast-prophet-neural-pro)

