from sklearn.metrics import accuracy_score, classification_report
from typing import Union, Tuple

import pandas as pd
import mlflow


class ModelEvaluator:
    def __init__(self, pipeline, X_test: Union[pd.DataFrame, None], y_test: Union[pd.Series, None]):
        self.pipeline = pipeline
        self.X_test = X_test
        self.y_test = y_test

    def evaluate_model(self) -> Tuple[Union[float, None], Union[str, None]]:
        try:
            if self.X_test is None or self.y_test is None:
                raise ValueError("X_test and/or y_test is None. Data not provided for evaluation.")
            
            y_pred = self.pipeline.predict(self.X_test['Message'])
            accuracy = accuracy_score(self.y_test, y_pred)
            report = classification_report(self.y_test, y_pred)
            # Log accuracy as a metric
            # mlflow.log_metrics({
            #     "accuracy": accuracy
            # })
            
            return accuracy, report
        except Exception as e:
            raise ValueError(f"Error during model evaluation: {str(e)}")

if __name__ == "__main__":
    X_test = None  # Replace with your test data
    y_test = None  # Replace with your test labels
    # model_evaluator = ModelEvaluator(train_model.pipeline, X_test, y_test)
    # accuracy, report = model_evaluator.evaluate_model()
    mlflow.end_run() 