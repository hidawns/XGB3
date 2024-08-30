import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from xgboost import XGBRegressor
import joblib

# Load the trained model and scaler
model = joblib.load('xgboost_model.pkl')
scaler = joblib.load('scaler (1).pkl')

# Streamlit app
st.title('Doctor Consultation Fee Prediction')

# Input fields
experience = st.number_input('Years of Experience', min_value=0, max_value=66, value=0)
rating = st.number_input('Doctor Rating', min_value=1, max_value=100, value=1)
num_of_qualifications = st.number_input('Number of Qualifications', min_value=1, max_value=10, value=1)

miscellaneous_info = st.selectbox('Miscellaneous Info Existent', ['Not Present', 'Present'])
miscellaneous_info = 1 if miscellaneous_info == 'Present' else 0

profile = st.selectbox('Doctor Specialization', ['Ayurveda', 'Dentist', 'Dermatologist', 'ENT Specialist', 'General Medicine', 'Homeopath'])
profile_mapping = {'Ayurveda': 0, 'Dentist': 1, 'Dermatologist': 2, 'ENT Specialist': 3, 
                       'General Medicine': 4, 'Homeopath': 5}
profile = profile_mapping[profile]

place = st.selectbox('Place', ['Bangalore',  'Chennai', 'Coimbatore', 'Delhi', 'Ernakulam', 'Hyderabad', 'Mumbai', 'Thiruvananthapuram', 'Unknown'])
place_mapping = {'Bangalore': 0, 'Chennai': 1, 'Coimbatore': 2, 'Delhi': 3, 'Ernakulam': 4, 
                     'Hyderabad': 5, 'Mumbai': 6, 'Thiruvananthapuram': 7, 'Unknown': 8}
place = place_mapping[place]

fee_category = 0.0
fee_category_mapping = {0.0 : 0}
fee_category = fee_category_mapping[fee_category]

# Create input DataFrame
input_data = pd.DataFrame({
    'Experience': [experience],
    'Rating': [rating],
    'Num_of_Qualifications': [num_of_qualifications],
    'Miscellaneous_Info': [miscellaneous_info],
    'Profile': [profile],
    'Place': [place],
    'Fee_category': [fee_category]
})

# Handle scaling
input_data['Experience'] = np.sqrt(input_data['Experience'])
input_data = scaler.transform(input_data)

# Predict fee
predicted_fee = model.predict(input_data)[0]

# Display result
st.write(f'Predicted Doctor Consultation Fee: ${predicted_fee:.2f}')
