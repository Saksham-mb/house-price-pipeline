from pathlib import Path
import joblib
import numpy as np
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
from data import DataExtractor
import json
import matplotlib.pyplot as plt

PROJECT_ROOT=Path(__file__).resolve().parent.parent
RESULTS_DIR=PROJECT_ROOT / 'results'
FIGURES_DIR=RESULTS_DIR / 'figures'

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

data_loader = DataExtractor()
X_test, y_test_log = data_loader.test_data()

model_path = PROJECT_ROOT / 'models' / 'xgb_best_pipeline.joblib'
pipeline = joblib.load(model_path)

y_pred_log = pipeline.predict(X_test)
rmse_log = root_mean_squared_error(y_test_log, y_pred_log)
r2_log = r2_score(y_test_log, y_pred_log)
print(f"RMSE on log transformed y is {rmse_log:.4f} and log R_2 score is {r2_log:.4f}")

y_test_actual = np.expm1(y_test_log)
y_pred_actual = np.expm1(y_pred_log)
rmse_dollars = root_mean_squared_error(y_test_actual, y_pred_actual)
mae_dollars = mean_absolute_error(y_test_actual, y_pred_actual)
r2_dollars = r2_score(y_test_actual, y_pred_actual)
print(f"Actual RMSE is {rmse_dollars:.2f}, MAE is {mae_dollars:.4f} and actual R_2 score is {r2_dollars:.4f}")

metrics={
    "test_rmsle": round(rmse_log, 4),
    "test_log_r2": round(r2_log, 4),
    "test_dollar_rmse": round(rmse_dollars, 2),
    "test_dollar_mae": round(mae_dollars, 2),
    "test_dollar_r2": round(r2_dollars, 4)
}

with open(RESULTS_DIR / 'metrics.json', 'w') as f:
    json.dump(metrics, f, indent=4)

plt.figure(figsize=(10, 8))
plt.plot([0, max(y_test_actual)], [0, max(y_test_actual)], linestyle='--', c='black')
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.scatter(y_test_actual, y_pred_actual)
plt.title("Actual Price vs Predicted Price")
plt.grid(True, alpha=0.7)
plt.savefig(FIGURES_DIR / "actual_vs_predicted_price.png", dpi=300, bbox_inches="tight")
plt.close()

plt.figure(figsize=(10, 8))
plt.scatter(y_pred_actual, y_test_actual-y_pred_actual, c='crimson')
plt.xlabel("Predicted Prices")
plt.ylabel("Residuals")
plt.title("Predicted Price vs Residuals")
plt.grid(True, alpha=0.7)
plt.savefig(FIGURES_DIR / "predicted_price_vs_residuals.png", dpi=300, bbox_inches="tight")
plt.close()