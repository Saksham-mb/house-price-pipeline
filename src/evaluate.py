from pathlib import Path
import joblib
import numpy as np
from sklearn.metrics import root_mean_squared_error
from data import DataExtractor
import matplotlib.pyplot as plt

PROJECT_ROOT=Path(__file__).resolve().parent.parent
RESULTS_DIR=PROJECT_ROOT / 'results'

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

data_loader = DataExtractor()
X_test, y_test_log = data_loader.test_data()

model_path = PROJECT_ROOT / 'models' / 'xgb_best_pipeline.joblib'
pipeline = joblib.load(model_path)

y_pred_log = pipeline.predict(X_test)
rmse_log = root_mean_squared_error(y_test_log, y_pred_log)
print(f"RMSE on log transformed y is {rmse_log:.4f}")

y_test_actual = np.expm1(y_test_log)
y_pred_actual = np.expm1(y_pred_log)
rmse_dollars = root_mean_squared_error(y_test_actual, y_pred_actual)
print(f"Actual RMSE is {rmse_dollars:.2f}")

plt.figure(figsize=(10, 8))
plt.plot([0, max(y_test_actual)], [0, max(y_test_actual)], linestyle='--', c='black')
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.scatter(y_test_actual, y_pred_actual)
plt.title("Actual Price vs Predicted Price")
plt.grid(True, alpha=0.7)
plt.savefig(RESULTS_DIR / "actual_vs_predicted_price.png", dpi=300, bbox_inches="tight")
plt.close()