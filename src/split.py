import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple, Union

class DataSplitting:
    def __init__(self, df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
        self.df = df
        self.test_size = test_size
        self.random_state = random_state
        self.X_train: Union[pd.DataFrame, None] = None
        self.X_test: Union[pd.DataFrame, None] = None
        self.y_train: Union[pd.Series, None] = None
        self.y_test: Union[pd.Series, None] = None

    def split_data(self) -> Tuple[Union[pd.DataFrame, None], Union[pd.DataFrame, None], Union[pd.Series, None], Union[pd.Series, None]]:
        try:
            if 'Message' not in self.df.columns or 'Category' not in self.df.columns:
                raise ValueError("Required columns ('Message' and 'Category') not found in the DataFrame")

            X = self.df[['Message']]
            y = self.df[['Category']]
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)
            return self.X_train, self.X_test, self.y_train, self.y_test
        except Exception as e:
            raise ValueError(f"Error during data splitting: {str(e)}")
