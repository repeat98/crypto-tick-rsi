import numpy as np
import pandas as pd

df = pd.read_csv("/Users/jannikassfalg/coding/crypto-tick-rsi/ml/labelsInd/SOLUSD_PERP-trades-2025-04_labels.csv")
rets = df["label"].to_numpy()
baseline_mse = np.var(rets)
target_good  = 0.8 * baseline_mse# ≈ 6.72e-6
target_great = 0.7 * baseline_mse#≈ 5.88e-6
print("Baseline MSE (zero predictor):", baseline_mse)
print("Target good MSE:", target_good) 
print("Target great MSE:", target_great)