import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from xgboost import XGBRegressor
import joblib

# Load the trained model and scaler
model = joblib.load('xgb_model.pkl')
scaler = joblib.load('scaler.pkl')
encoder = joblib.load('encoder.pkl')

# Streamlit app
st.title('Doctor Consultation Fee Prediction')

# Input fields
experience = st.number_input('Years of Experience', min_value=0, max_value=66, value=0)
num_of_qualifications = st.number_input('Number of Qualifications', min_value=1, max_value=10, value=1)
rating = st.number_input('Doctor Rating', min_value=1, max_value=100, value=1)
miscellaneous_info = st.selectbox('Miscellaneous Info Existent', ['Not Present', 'Present'])
profile = st.selectbox('Doctor Specialization', ['Ayurveda', 'Dentist', 'Dermatologist', 'ENT Specialist', 'General Medicine', 'Homeopath'])
place = st.selectbox('Place', ['Bangalore',  'Chennai', 'Coimbatore', 'Delhi', 'Ernakulam', 'Hyderabad', 'Mumbai', 'Thiruvananthapuram', 'Other'])

# Encoding and scaling
input_data = pd.DataFrame({
    'Experience': [experience],
    'Rating': [rating],
    'Place': [place],
    'Profile': [profile],
    'Miscellaneous_Info': [miscellaneous_info],
    'Num_of_Qualifications': [num_of_qualifications],
    'Fee_category': [0.0]  # Default value for Fee_category
})

# Transform data
input_data['Experience'] = np.sqrt(input_data['Experience'])
input_data = pd.get_dummies(input_data, columns=['Place', 'Profile', 'Miscellaneous_Info'], drop_first=True)
input_data = scaler.transform(input_data)

# Predict fee
predicted_fee = model.predict(input_data)[0]

# Display result
st.write(f'Predicted Doctor Consultation Fee: {predicted_fee:.2f}')
