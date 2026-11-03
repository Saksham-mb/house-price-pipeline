import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, TargetEncoder


class MasVnrCleaner(BaseEstimator, TransformerMixin):
  def fit(self, X, y=None):
    temp = X["MasVnrType"].fillna("None")
    valid_types = temp[temp != "None"]
    self.real_veneer_mode_ = (valid_types.mode()[0] if not valid_types.empty else "BrkFace")
    return self

  def transform(self, X):
    X_copy = X.copy()
    X_copy["MasVnrType"] = X_copy["MasVnrType"].fillna("None")
    mismatch_mask = (X_copy["MasVnrType"] == "None") & (X_copy["MasVnrArea"] > 0)
    X_copy.loc[mismatch_mask, "MasVnrType"] = self.real_veneer_mode_
    return X_copy


class LotFrontageImputer(BaseEstimator, TransformerMixin):
  def __init__(self, group_col="Neighborhood", target_col="LotFrontage"):
    self.group_col = group_col
    self.target_col = target_col
    self.group_median_ = {}
    self.global_median_ = None

  def fit(self, X, y=None):
    self.group_median_ = (X.groupby(self.group_col)[self.target_col].median().to_dict())
    self.global_median_ = X[self.target_col].median()
    return self

  def transform(self, X, y=None):
    X_copy = X.copy()
    mask = X_copy[self.target_col].isna()
    X_copy.loc[mask, self.target_col] = (X_copy.loc[mask, self.group_col].map(self.group_median_).fillna(self.global_median_))
    return X_copy


def build_preprocessor():
  low_variance_cols = ["Utilities", "Street", "PoolArea"]
  na_is_zero = SimpleImputer(strategy="constant", fill_value=0)

  electrical_pipeline = Pipeline(steps=[
          ("impute", SimpleImputer(strategy="most_frequent")),
          ("encode",OneHotEncoder(dtype="int64", handle_unknown="infrequent_if_exist", min_frequency=12, sparse_output=False))
          ])

  na_zero_qc_cols = ["BsmtQual","BsmtCond","GarageQual","FireplaceQu","GarageCond","PoolQC"]
  na_zero_qc_map = ["None", "Po", "Fa", "TA", "Gd", "Ex"]
  na_zero_qc_pipeline = Pipeline(steps=[
          ("impute", SimpleImputer(strategy="constant", fill_value="None", keep_empty_features=True)),
          ("encode",OrdinalEncoder(categories=[na_zero_qc_map]*len(na_zero_qc_cols), dtype="int64", handle_unknown="use_encoded_value", unknown_value=-1))
          ])

  na_zero_BsmtExp_cols = ["BsmtExposure"]
  na_zero_BsmtExp_map = ["None", "No", "Mn", "Av", "Gd"]
  na_zero_BsmtExp_pipeline = Pipeline(steps=[
          ("impute", SimpleImputer(strategy="constant", fill_value="None")),
          ("encode", OrdinalEncoder(categories=[na_zero_BsmtExp_map] * len(na_zero_BsmtExp_cols), dtype="int64", handle_unknown="use_encoded_value", unknown_value=-1))
          ])

  na_zero_BsmtFinType_cols = ["BsmtFinType1", "BsmtFinType2"]
  na_zero_BsmtFinType_map = ["None", "Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"]
  na_zero_BsmtFinType_pipeline = Pipeline(steps=[
          ("impute", SimpleImputer(strategy="constant", fill_value="None")),
          ("encode", OrdinalEncoder(categories=[na_zero_BsmtFinType_map] * len(na_zero_BsmtFinType_cols), dtype="int64", handle_unknown="use_encoded_value", unknown_value=-1))
          ])

  na_zero_GarageFinish_col = ["GarageFinish"]
  na_zero_GarageFinish_map = ["None", "Unf", "RFn", "Fin"]
  na_zero_GarageFinish_pipeline = Pipeline(steps=[
          ("impute", SimpleImputer(strategy="constant", fill_value="None")),
          ("encode", OrdinalEncoder(categories=[na_zero_GarageFinish_map] * len(na_zero_GarageFinish_col), dtype="int64", handle_unknown="use_encoded_value", unknown_value=-1))
          ])

  na_zero_Fence_col = ["Fence"]
  na_zero_Fence_map = ["None", "MnWw", "GdWo", "MnPrv", "GdPrv"]
  na_zero_Fence_pipeline = Pipeline(steps=[
          ("impute", SimpleImputer(strategy="constant", fill_value="None")),
          ("encode", OrdinalEncoder(categories=[na_zero_Fence_map] * len(na_zero_Fence_col), dtype="int64", handle_unknown="use_encoded_value", unknown_value=-1))
          ])

  categorical_data_tbohe = ["MSZoning", "Alley", "LotConfig", "Condition1", "BldgType", "RoofStyle", "MasVnrType", "Foundation", "GarageType", "SaleType", "SaleCondition", "Condition2", "RoofMatl", "Heating", "MiscFeature"]
  ohe_pipeline = Pipeline(steps=[
          ("impute_none", SimpleImputer(strategy="constant", fill_value="None")),
          ("encode", OneHotEncoder(dtype="int64", handle_unknown="infrequent_if_exist", min_frequency=12, sparse_output=False))
          ])

  categorical_data_tbtrgte = ["MSSubClass", "Neighborhood", "Exterior1st", "Exterior2nd", "HouseStyle"]
  trgte = TargetEncoder(target_type="continuous", cv=10, random_state=42)

  return ColumnTransformer([
          ("electrical", electrical_pipeline, ["Electrical"]),
          ("masvnrarea", na_is_zero, ["MasVnrArea", "GarageYrBlt"]),
          ("drop_low_var", "drop", low_variance_cols),
          ("na_qc", na_zero_qc_pipeline, na_zero_qc_cols),
          ("BsmtExp", na_zero_BsmtExp_pipeline, na_zero_BsmtExp_cols),
          ("BsmtFinType", na_zero_BsmtFinType_pipeline, na_zero_BsmtFinType_cols),
          ("GarageFinish", na_zero_GarageFinish_pipeline, na_zero_GarageFinish_col),
          ("Fence", na_zero_Fence_pipeline, na_zero_Fence_col),
          ("PavedDrive",OrdinalEncoder(categories=[["N", "P", "Y"]],dtype="int64",handle_unknown="use_encoded_value",unknown_value=-1),["PavedDrive"]),
          ("Functional",OrdinalEncoder(categories=[["Sal", "Sev", "Maj2", "Maj1", "Mod", "Min2", "Min1", "Typ"]],dtype="int64",handle_unknown="use_encoded_value",unknown_value=-1),["Functional"]),
          ("CentralAir",OrdinalEncoder(categories=[["N", "Y"]],dtype="int64",handle_unknown="use_encoded_value",unknown_value=-1),["CentralAir"]),
          ("LotShape",OrdinalEncoder(categories=[["IR3", "IR2", "IR1", "Reg"]],dtype="int64",handle_unknown="use_encoded_value",unknown_value=-1),["LotShape"]),
          ("LandContour",OrdinalEncoder(categories=[["Low", "HLS", "Bnk", "Lvl"]],dtype="int64",handle_unknown="use_encoded_value",unknown_value=-1),["LandContour"]),
          ("LandSlope",OrdinalEncoder(categories=[["Sev", "Mod", "Gtl"]],dtype="int64",handle_unknown="use_encoded_value",unknown_value=-1),["LandSlope"]),
          ("qc",OrdinalEncoder(categories=[["Po", "Fa", "TA", "Gd", "Ex"]] * 4,dtype="int64",handle_unknown="use_encoded_value",unknown_value=-1),["KitchenQual", "ExterQual", "HeatingQC", "ExterCond"]),
          ("ohe", ohe_pipeline, categorical_data_tbohe),
          ("trgte", trgte, categorical_data_tbtrgte),
      ],remainder="passthrough")