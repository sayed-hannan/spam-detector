import re
import string
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import LabelEncoder  # For categorical encoding

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

import warnings
from sklearn.exceptions import DataConversionWarning

# ... your code ...
# Suppress the DataConversionWarning
warnings.filterwarnings("ignore", category=DataConversionWarning)



class DataPreprocessing:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def preprocess_email(self, email_text: str) -> str:
        try:
            email_text = email_text.lower()
            email_text = re.sub(r'\d+', '', email_text)
            email_text = email_text.translate(str.maketrans('', '', string.punctuation))
            words = word_tokenize(email_text)
            stop_words = set(stopwords.words('english'))
            words = [word for word in words if word not in stop_words]
            lemmatizer = nltk.stem.WordNetLemmatizer()
            words = [lemmatizer.lemmatize(word) for word in words]
            cleaned_text = ' '.join(words)
            return cleaned_text
        except Exception as e:
            raise ValueError(f"Error during email preprocessing: {str(e)}")

    def encode_categorical_variables(self):
        try:
            if 'Category' in self.df.columns:
                label_encoder = LabelEncoder()
                self.df['Category'] = label_encoder.fit_transform(self.df['Category'].values.ravel())
            else:
                raise ValueError("No 'Category' column found in the DataFrame")
        except Exception as e:
            raise ValueError(f"Error during categorical variable encoding: {str(e)}")


    def preprocess_data(self):
        try:
            if 'Category' not in self.df.columns or 'Message' not in self.df.columns:
                raise ValueError("Required columns ('Category' and 'Message') not found in the DataFrame")

            self.encode_categorical_variables()  # Encode categorical variables
            self.df['Message'] = self.df['Message'].apply(self.preprocess_email)
        except Exception as e:
            raise ValueError(f"Error during data preprocessing: {str(e)}")
