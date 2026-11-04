# Ames Housing Price Prediction — Modular XGBoost & Custom Scikit-Learn Pipeline

An end-to-end machine learning pipeline for the Ames Housing dataset. Rather than performing DataFrame mutations in a notebook, this repository showcases domain specific data cleaning, group imputation, categorical encoding, and XGBoost Tree inside a leak-free `scikit-learn-Pipeline` with `joblib` for saving and loading the pipeline.

---

## Project Structure

```text
house-price-pipeline/
├── datasets/
│   ├── House-Prices-Data/
│   │   └── train.csv
│   └── README.md
├── models/
│   └── xgb_best_pipeline.joblib
├── notebooks/
│   └── 1_EDA.ipynb
├── results/
│   ├── figures/
│   │   ├── actual_vs_predicted_price.png
│   │   └── predicted_price_vs_residuals.png
│   └── metrics.json
├── src/
│   ├── data.py
│   ├── features.py
│   ├── model.py
│   ├── train.py
│   └── evaluate.py
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

---

## Engineering Architecture & Feature Pipeline

### 1. Structural Outlier Filtering & Target Transformation (`src/data.py`)
* **Outlier Removal:** Drops the two documented structural outliers (`Id` 524 and 1299) prior to partitioning.
* **Target Normalization:** Applies $\log(1 + y)$ (`numpy.log1p`) to right-skewed `SalePrice` targets during training and exponentiates predictions during evaluation to reduce error variance in `SalePrice`.

### 2. Custom Stateful Scikit-Learn Transformers (`src/features.py`)
All transformations inherit from `BaseEstimator` and `TransformerMixin` and learn statistics strictly from training folds during cross-validation:
* **`MasVnrCleaner`:** Resolves logical data inconsistencies where `MasVnrType` is recorded as `'None'` despite `MasVnrArea > 0`, imputing the learned training-fold mode of valid masonry veneer types.
* **`LotFrontageImputer`:** Performs group-based median imputation for `LotFrontage` values grouped by `Neighborhood`, with fallback to the global training median.
* **Multi-Modal `ColumnTransformer`:**
  * **Domain Null Encoding:** Maps structural `NaN` values (representing the physical absence of a basement, garage, pool, or fence) to an explicit `'None'` category before applying ordered `OrdinalEncoder` hierarchies (`None < Po < Fa < TA < Gd < Ex`).
  * **Target Encoding:** Applies 10-fold out-of-fold continuous `TargetEncoder` to multi-class features (`Neighborhood`, `MSSubClass`, `Exterior1st`, `Exterior2nd`, `HouseStyle`) without target leakage.
  * **One-Hot Encoding:** Applies `OneHotEncoder(min_frequency=12, handle_unknown='infrequent_if_exist')` to nominal features to group rare categories and prevent dimensionality explosion.
  * **Low Variance Features & Fallback Imputation:** Drops near-zero variance noise columns (`Utilities`, `Street`, `PoolArea`) and wraps remaining continuous features in `SimpleImputer(strategy='median')`.

### 3. Gradient Boosting (`src/model.py`)
Couples the preprocessing pipeline with a regularized `XGBRegressor` (`tree_method='hist'`, `n_estimators=1500`, `learning_rate=0.03`, `max_depth=4`, `min_child_weight=2`, `subsample=0.65`, `colsample_bytree=0.6`, `reg_lambda=8.0`, `reg_alpha=1.0`) validated via 5-fold shuffled `KFold` cross-validation.

---

## Holdout Test Set Performance (`X_test`)

Evaluated on the untouched 20% holdout test partition (`results/metrics.json`):

| Metric | Score |
| :--- | :--- |
| **Test RMSLE (Log RMSE)** | **0.1256** |
| **Test Log $R^2$ Score** | **0.9064** |
| **Test Dollar $R^2$ Score** | **0.9238** |
| **Test Dollar MAE** | **$14,555.61** |
| **Test Dollar RMSE** | **$20,516.07** |

Predicted price against it's real price (`actual_vs_predicted_price.png`) and residual distribution against predicted price (`predicted_price_vs_residuals.png`) plots are exported to `results/figures/`.

---

## Setup & Usage

### 1. Installation
Clone the repository and install the required dependencies:
```bash
pip install -r requirements.txt
```

### 2. Dataset Preparation
Download the [House Prices - Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data) dataset and place `train.csv` inside `datasets/House-Prices-Data/`.

### 3. Train & Serialize the Pipeline
Run 5-fold cross-validation on the training split and save the fitted end-to-end pipeline artifact to `models/xgb_best_pipeline.joblib`:
```bash
python src/train.py
```

### 4. Evaluate on Holdout Test Set
Load the serialized pipeline, score the holdout test set, export `results/metrics.json`, and generate diagnostic plots in `results/figures/`:
```bash
python src/evaluate.py
```
