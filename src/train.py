from pathlib import Path
import joblib
from data import DataExtractor
from model import models

PROJECT_ROOT=Path(__file__).resolve().parent.parent
MODEL_DIR=PROJECT_ROOT / 'models'
MODEL_DIR.mkdir(parents=True, exist_ok=True)

data_loader=DataExtractor()
X_train, y_log_train=data_loader.train_data()

xgb_model=models()
model_0, rmse=xgb_model.model_pipeline(X_train, y_log_train)
print(f'RMSE on log y is {rmse:.4f}')

MODEL_PATH=MODEL_DIR / 'xgb_best_pipeline.joblib'
joblib.dump(model_0, MODEL_PATH)
print(f"Saved the best estimator")