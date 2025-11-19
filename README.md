# Apple Stock Time-Series Forecasting

Daily closing prices for Apple (`AAPL.csv`) are modeled with several deep-learning architectures and a Prophet baseline to understand how different sequence models capture price dynamics. Each notebook in this repo corresponds to one of the experiments described below.

## Dataset
- Source: Kaggle Apple share price data set (same schema as [apple_share_price.csv](https://www.kaggle.com/code/ramjasmaurya/apple-stock-price-prediction-using-lstm/input?select=apple_share_price.csv)).
- Features used: `date`, `close` (renamed to `stock_price`).
- Coverage: 27 May 2015 – 22 May 2020.
- Splits (identical across notebooks):
  - Train: 27 May 2015 – 22 May 2018
  - Validation: 23 May 2018 – 22 May 2019
  - Test: 23 May 2019 – 22 May 2020
- Pre-processing:
  - Records indexed by timestamp and sorted chronologically.
  - Min–Max scaling to `[0, 1]`.
  - Sliding-window datasets built with 12 historical timesteps (≈ 12 trading days) to predict future prices.

## Repository contents

| File | Description |
| --- | --- |
| `stacked_LSTM.ipynb` | Two-layer stacked LSTM regression model (128 + 32 units) trained with Keras/TensorFlow. |
| `stacked_GRU.ipynb` | Mirror of the LSTM notebook but with stacked GRU layers. |
| `seq2seq_LSTM.ipynb` | Encoder–decoder LSTM that consumes 12 days and predicts the next 6 steps jointly. |
| `prophet.ipynb` | Facebook/Meta Prophet additive seasonal baseline. |
| `model.png` | Saved network diagram exported via `plot_model`. |
| `trained_model.h5`| Checkpoints for the stacked LSTM experiment. |

## Modeling workflow (common steps)
- Load `AAPL.csv`, parse timestamps, and visualize raw vs normalized prices.
- Use helper utilities (`create_dataset`, `split_train_valid_test`) to produce supervised learning tensors plus aligned date indices for evaluation/plotting.
- Train networks with Adam, early stopping, and model checkpoints; best weights are reloaded before evaluation.
- Convert normalized predictions back to the original scale by re-adding the first value of each window (`start_values` in the notebooks).
- Report RMSE and MAE for train/validation/test splits, both for 1-step ahead and, in the seq2seq case, for the full decoder horizon.

## Experiment results
All metrics are pulled directly from the notebooks listed above. Values are on the original (unscaled) price axis because denormalization is applied before evaluation.

| Model | Forecast horizon | Train (RMSE / MAE) | Valid (RMSE / MAE) | Test (RMSE / MAE) |
| --- | --- | --- | --- | --- |
| Stacked LSTM (`stacked_LSTM.ipynb`) | t + 1 step | 1.91 / 1.34 | 3.61 / 2.51 | **6.48 / 4.31** |
| Stacked GRU (`stacked_GRU.ipynb`) | t + 1 step | 1.99 / 1.41 | 3.58 / 2.58 | **6.53 / 4.09** |
| Seq2Seq LSTM (`seq2seq_LSTM.ipynb`) | t + 1 step (decoder index 0) | 2.56 / 1.83 | 4.15 / 3.22 | **7.03 / 4.66** |
| Seq2Seq LSTM (`seq2seq_LSTM.ipynb`) | t + 6 step (decoder index 5) | 5.57 / 4.14 | 7.69 / 5.99 | **12.98 / 9.22** |
| Prophet (`prophet.ipynb`) | t + 1 step | 9.57 / 7.23 | — | **91.89 / 76.71** |

Key takeaways:
- Switching from LSTM to GRU delivers similar accuracy with slightly fewer parameters.
- The seq2seq decoder maintains stable gradients for multi-step forecasts but accumulates error on longer horizons, as expected.
- Prophet underfits the strong upward trend in the later years, leading to significantly larger errors despite low training loss.

## How to run the notebooks
1. Create a Python 3.11+ environment and install the dependencies used in the notebooks:
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn tensorflow==2.* prophet
   ```
2. Open the notebook of interest in JupyterLab/VScode/Cursor (`jupyter lab stacked_LSTM.ipynb` etc.).
3. Ensure that `AAPL.csv` is located at `../AAPL.csv` relative to the notebooks (already configured in the code).
4. Run all cells to reproduce the metrics above or tweak hyperparameters/window sizes for new experiments.

---
Maintainer: Priyanka Jagadala, 2025.
