# China Galaxy Securities — Hybrid LSTM–XGBoost Quantitative Research

> Quantitative research project completed during a summer internship on the
> Quantitative Research desk at **China Galaxy Securities Co., Ltd.**
> (中国银河证券), Shanghai, July – September 2024.

This repository contains the Jupyter notebook and the visual reports
produced as part of a research project on **hybrid sequence + tabular
modeling** for high-frequency equity return prediction. The notebook
combines an **LSTM** model (for temporal dependence in price sequences)
with an **XGBoost** model (for tabular feature interactions) into a
single hybrid pipeline, validated against a single-name equity time
series.

The work was undertaken as part of [Colar Wang](https://colar-wiki.vercel.app/wiki/Colar_Wang/)'s
2024 internship and is documented in the corresponding
[Colarpedia article on China Galaxy Securities](https://colar-wiki.vercel.app/wiki/China_Galaxy_Securities/).

---

## Repository contents

```
.
├── Hybrid_Approach.ipynb        # main notebook (104 cells, EDA → features → XGB → LSTM → hybrid)
├── reports/                     # rendered PDF reports from each modeling stage
│   ├── 1.eda analysis.pdf       # exploratory data analysis
│   ├── 2.apple cdf长期收益率.pdf # long-term return CDF
│   ├── 3.时间特征.pdf           # time-of-day / day-of-week features
│   ├── 4.K线核心数值.pdf        # OHLCV-derived candlestick features
│   ├── 5.xgboost window=2.pdf   # XGBoost — rolling window 2
│   ├── 5.xgboost window=5.pdf   # XGBoost — rolling window 5
│   ├── 5.xgboost window=10.pdf  # XGBoost — rolling window 10
│   ├── 5.xgboost window=20.pdf  # XGBoost — rolling window 20
│   ├── 6.feature importance.pdf # XGBoost feature importance
│   ├── 7.xgboost 模型对比linear regression和true return.pdf  # baseline comparison
│   ├── 7.xgb.png                # XGBoost prediction overlay
│   ├── 8.lstm training set.pdf  # LSTM in-sample fit
│   ├── 8.lstm 的testing set.pdf # LSTM out-of-sample fit
│   └── 9.最终预测.pdf            # final hybrid prediction
└── README.md
```

---

## Methodology

The notebook follows a six-stage research pipeline:

1. **EDA.** Distributional analysis of returns, autocorrelation
   structure, and regime-aware volatility profiling on a single-name
   equity time series.
2. **Feature engineering.** Derivation of time-of-day, day-of-week,
   and OHLCV candlestick features at multiple rolling-window scales
   (2, 5, 10, 20).
3. **XGBoost modeling.** Tabular gradient-boosting return prediction
   across the four window sizes, with rolling walk-forward
   cross-validation.
4. **Feature importance.** SHAP-style decomposition of which
   engineered features dominate the XGBoost decision surface.
5. **LSTM modeling.** Sequence-to-one return prediction trained on
   the same windowed feature set, with separate train/test splits.
6. **Hybrid ensemble.** Combination of XGBoost and LSTM predictions
   into a single forecast, evaluated against true returns and a
   linear-regression baseline.

A simulated backtest of the final hybrid pipeline reported an
**approximately 50 percent annualized return** under a high-frequency
signal-generation regime — the figure cited in the corresponding
[Colarpedia entry](https://colar-wiki.vercel.app/wiki/China_Galaxy_Securities/).

---

## Stack

- **Python 3** (Jupyter notebook)
- `pandas`, `numpy`, `scipy.stats` — data and statistics
- `yfinance` — market data ingestion
- `matplotlib`, `seaborn` — visualization
- `scikit-learn` — preprocessing, baselines, validation
- `xgboost` — gradient-boosting model
- `keras` / `tensorflow` — LSTM model

---

## Reproduction

```bash
git clone https://github.com/wangxuzhou666-arch/china-galaxy-securities-quant
cd china-galaxy-securities-quant
pip install pandas numpy scipy yfinance matplotlib seaborn scikit-learn xgboost tensorflow
jupyter lab Hybrid_Approach.ipynb
```

The notebook downloads its source data via `yfinance` at runtime and
does not require any external data files.

---

## Data and confidentiality

This repository contains **only publicly available market data** sourced
at runtime from Yahoo Finance via the `yfinance` Python package. It
contains:

- **No** proprietary or non-public market data
- **No** China Galaxy Securities internal data, customer data, account
  data, or trading positions
- **No** internal alpha signals, internal models, or internal IP belonging
  to China Galaxy Securities or any third party

The notebook is published as a personal research artifact written by
the author during the internship period. The methodology — hybrid
sequence + tabular modeling on equity time series — is a standard
research pattern documented in the public quantitative-finance
literature, and was independently re-implemented by the author against
public data. Nothing in this repository should be construed as
representing the views, methods, or proprietary work of China Galaxy
Securities Co., Ltd.

## Status

Archived as a research artifact. The pipeline is not maintained as a
live trading system. All return figures are simulated backtests on
historical data and **must not** be interpreted as forward-looking
performance projections.

## Author

[Colar Wang (王旭洲)](https://www.linkedin.com/in/xuzhou-wang/) —
Quantitative Research Intern, China Galaxy Securities, 2024.
