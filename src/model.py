from xgboost import XGBRegressor
from features import MasVnrCleaner, LotFrontageImputer, build_preprocessor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

class models:
    def __init__(self):
        self.params_grid = {
            'max_depth': 4,
            'min_child_weight': 2,
            'learning_rate': 0.03,
            'n_estimators': 1500,
            'subsample': 0.65,
            'colsample_bytree': 0.6,
            'reg_lambda': 8.0,
            'reg_alpha': 1.0,
            'gamma': 0.0
        }
    def build_and_evaluate(self, pipeline, X, y):
            scores = cross_val_score(pipeline, X, y, cv=5, scoring="neg_root_mean_squared_error", n_jobs=-1)
            cv_rmse = scores.mean()
            pipeline.fit(X, y)
            return pipeline, -cv_rmse
    def model_pipeline(self, X, y):
        pipeline = Pipeline(steps=[
            ('masvnrcleaner', MasVnrCleaner()),
            ('lotfrontageimputer', LotFrontageImputer(group_col='Neighborhood', target_col='LotFrontage')),
            ('preprocessing', build_preprocessor()),
            ('model', XGBRegressor(**self.params_grid, tree_method='hist',random_state=42,n_jobs=-1))
        ])
        
        return self.build_and_evaluate(pipeline, X, y)