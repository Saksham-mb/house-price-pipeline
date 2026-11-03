from pathlib import Path
from sklearn.model_selection import train_test_split
import pandas as pd
from numpy import log1p

PROJECT_ROOT=Path(__file__).parent.parent
DATASET_DIR=PROJECT_ROOT / 'datasets' / 'House-Prices-Data' / 'train.csv'


class DataExtractor:
    def __init__(self, DATASET_DIR: Path = DATASET_DIR):
        self.df=pd.read_csv(DATASET_DIR)
        self.df=self.df.drop([524, 1299])
        self.X = self.df.drop(columns=['SalePrice', 'Id'], errors='ignore')
        self.y=self.df['SalePrice']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.3, random_state=42)
    def train_data(self):
        self.y_train_log=log1p(self.y_train)
        return self.X_train, self.y_train_log
    def test_data(self):
        self.y_test_log=log1p(self.y_test)
        return self.X_test, self.y_test_log