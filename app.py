import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load model, scaler, and encoder
model = pickle.load(open('xgboost_model.pkl', 'rb'))
scaler = pickle.load(open('scaler (1).pkl', 'rb'))
encoder = pickle.load(open('encoder (1).pkl', 'rb'))

# Function to preprocess user inputs
def preprocess_inputs(experience, rating, place, profile, misc_info, num_of_qualifications):
    data = {
        'Experience': [experience],
        'Rating': [rating],
        'Place': [place],
        'Profile': [profile],
        'Miscellaneous_Info': [misc_info],
        'Num_of_Qualifications': [num_of_qualifications]
    }
    df = pd.DataFrame(data)
    
    # Encode categorical variables
    df['Place'] = encoder.transform(df['Place'])
    df['Profile'] = encoder.transform(df['Profile'])
    
    # Scale features
    df_scaled = scaler.transform(df)
    
    return df_scaled

# Streamlit app interface
st.title('Doctor Consultation Fee Prediction')

# User inputs
experience = st.number_input('Years of Experience', min_value=0, max_value=66)
rating = st.number_input('Doctor Rating', min_value=1, max_value=100)
place = st.selectbox('Place', ['Bangalore', 'Mumbai', 'Delhi', 'Hyderabad', 'Chennai', 'Coimbatore', 'Ernakulam', 'Thiruvananthapuram', 'Other'])
profile = st.selectbox('Doctor Specialization', ['Ayurveda', 'Dentist', 'Dermatologist', 'ENT Specialist', 'General Medicine', 'Homeopath'])
misc_info = st.selectbox('Miscellaneous Info Existent', ['Not Present', 'Present'])
num_of_qualifications = st.number_input('Number of Qualifications', min_value=1, max_value=10)

# Preprocess and predict
if st.button('Predict Fee'):
    # Encode 'Place' and 'Profile' correctly
    place_encoded = encoder.transform([place])[0]
    profile_encoded = encoder.transform([profile])[0]
    
    # Preprocess inputs
    inputs_scaled = preprocess_inputs(experience, rating, place_encoded, profile_encoded, misc_info, num_of_qualifications)
    
    # Make prediction
    predicted_fee = model.predict(inputs_scaled)[0]
    st.write(f'Predicted Fee: {predicted_fee:.2f}')
