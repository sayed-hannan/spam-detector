import streamlit as st
import pandas as pd
import joblib
from src.data_ingestion import DataIngestion

# Load your data
@st.cache_data
def load_data(data_file_path):
    data_ingestion = DataIngestion(data_file_path)
    df = data_ingestion.ingest_data()
    return df

data_file_path = './data/spam.csv'
df = load_data(data_file_path)

# Load the pre-trained model
@st.cache_resource
def load_model(model_path):
    return joblib.load(model_path)

model_path = './models/spam_model.pkl'
model = load_model(model_path)

# Sidebar with options
st.sidebar.title('Spam Detection App')

# Select a sample message for prediction
st.sidebar.subheader('Select a Sample Message')
selected_sample_index = st.sidebar.selectbox('Choose a sample:', df.index)

# Display selected sample message and prediction
st.title('Spam Detection App')
st.subheader('Selected Sample Message:')
selected_message = df.loc[selected_sample_index, 'Message']
st.write(selected_message)

# Function to predict spam
def predict_spam(text):
    prediction = model.predict([text])
    return prediction[0]

# Button to trigger prediction
if st.button('Predict'):
    prediction = predict_spam(selected_message)
    if prediction == 1:
        st.error('This message is predicted as spam.')
    else:
        st.success('This message is predicted as not spam.')

# Display some data samples (optional)
st.sidebar.subheader('Explore Data')
if st.sidebar.checkbox('Show Sample Data'):
    st.write(df.head(10))

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("Built with Streamlit")
