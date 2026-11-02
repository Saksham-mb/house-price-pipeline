from pathlib import Path
import joblib
import numpy as np
from sklearn.metrics import root_mean_squared_error
from data import DataExtractor

data_loader = DataExtractor()
X_test, y_test_log = data_loader.test_data()

model_path = Path(__file__).resolve().parent.parent / 'models' / 'xgb_best_pipeline.joblib'
pipeline = joblib.load(model_path)

y_pred_log = pipeline.predict(X_test)
rmse_log = root_mean_squared_error(y_test_log, y_pred_log)
print(f"RMSE on log transformed y is {rmse_log:.4f}")

y_test_actual = np.expm1(y_test_log)
y_pred_actual = np.expm1(y_pred_log)
rmse_dollars = root_mean_squared_error(y_test_actual, y_pred_actual)
print(f"Actual RMSE is {rmse_dollars:.2f}")