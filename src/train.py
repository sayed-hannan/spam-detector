from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from typing import Union
import pandas as pd




class ModelTrainer:
    def __init__(self, X_train: Union[pd.DataFrame, None], y_train: Union[pd.Series, None]):
        self.X_train = X_train
        self.y_train = y_train
        self.pipeline = self.create_pipeline()

    def create_pipeline(self) -> Pipeline:
        try:
            tfidf_vectorizer = TfidfVectorizer()
            classifier = MultinomialNB()
            return Pipeline([
                ('tfidf_vectorizer', tfidf_vectorizer),
                ('classifier', classifier)
            ])
        except Exception as e:
            raise ValueError(f"Error creating pipeline: {str(e)}")

    def train_model(self) -> None:
        try:
            if self.X_train is None or self.y_train is None:
                raise ValueError("X_train and/or y_train is None. Data not provided for training.")
            
            self.pipeline.fit(self.X_train['Message'], self.y_train)
            
        
            
        except Exception as e:
            raise ValueError(f"Error during model training: {str(e)}")

if __name__ == "__main__":
    # Load your data and create the ModelTrainer instance
    # Replace the None values with your data
    X_train = None  # Replace with your training data
    y_train = None  # Replace with your training labels
    model_trainer = ModelTrainer(X_train, y_train)
    
    # Train the model and log it with MLflow
    model_trainer.train_model()
