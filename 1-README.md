# Orange Problem - NIFTY50 Stock Buy/Sell Prediction (PyTorch TabNet)

Solution for the **Orange Problem** assignment from
[tiwvis/MLBD-Projects](https://github.com/tiwvis/MLBD-Projects/tree/main/orange_problem).

## Task

1. Load the stock data (`nifty50_p2`)
2. Feature engineering: `day, month, year, is_quarter_end, open-close, low-high, target`
3. Perform EDA
4. Train a PyTorch TabNet model for all the stocks
   - Prediction variables: **stock, open-close, low-high, is_quarter_end**
   - Predicted variable: **target**
5. Predict on `nifty50_prd2` and output in the specified format (Target = 1: Buy, Target = 0: Sell)

## Repository contents

| File | Description |
|---|---|
| `orange_problem_tabnet.ipynb` | Fully executed notebook: data loading, feature engineering, EDA, TabNet training, prediction |
| `orange_problem_tabnet.py` | Plain-Python export of the same notebook |
| `Orange_Problem_Analysis_and_Results.pdf` | Analysis and results report: EDA snapshots, result/output, interpretation |
| `prediction_output.csv` | Final Buy/Sell signal per stock in the specified output format |

## Data

- Training: [`nifty50_p2`](https://github.com/tiwvis/MLBD-Projects/blob/main/orange_problem/nifty50_p2) - 5 stocks
  (SBIN, ONGC, BEL, INFY, TCS), 4,931 trading days each, 16-10-2006 to 13-10-2026.
- Prediction: [`nifty50_prd2`](https://github.com/tiwvis/MLBD-Projects/blob/main/orange_problem/nifty50_prd2) - one row
  per stock for 14-10-2026.

## Results

- Single TabNet classifier trained for all 5 stocks (time-aware split: last 20% of each stock's
  timeline held out for validation; boundary training rows whose labels would use a
  validation-period close are purged from both sets).
- Validation accuracy **0.5034**, ROC AUC **0.5183** - next-day direction from same-day OHLC
  features is close to a coin flip, which the report interprets honestly.
- Top feature: `open-close` (38.9% importance).
- Predictions come from the same model that produced the validation metrics (no full-history refit).

### Final signals (14-10-2026)

| Stock | Signal |
|---|---|
| Bharat Electronics Limited | Sell |
| Infosys Limited | Sell |
| Oil & Natural Gas Corporation Limited | Sell |
| State Bank of India | Buy |
| Tata Consultancy Services Limited | Sell |

## Running

```
pip install torch pytorch-tabnet pandas numpy matplotlib seaborn scikit-learn
jupyter notebook orange_problem_tabnet.ipynb
```

The notebook downloads the data directly from the source repository at runtime.
