from data_ingestion import DataIngestion
from preprocess import DataPreprocessing
from split import DataSplitting
from train import ModelTrainer
from evaluate import ModelEvaluator

# Data Ingestion
data_file = "../data/spam.csv"
data_ingestion = DataIngestion(data_file)
df = data_ingestion.ingest_data()

# Data Preprocessing
data_preprocessing = DataPreprocessing(df)
data_preprocessing.preprocess_data()

# Data Splitting
data_splitting = DataSplitting(df)
X_train, X_test, y_train, y_test = data_splitting.split_data()

# Model Training
model_trainer = ModelTrainer(X_train, y_train)
model_trainer.train_model()

# Model Evaluation
model_evaluator = ModelEvaluator(model_trainer.pipeline, X_test, y_test)
accuracy, report = model_evaluator.evaluate_model()

print(f"Accuracy: {accuracy}")
print(report)
