import streamlit as st
import pickle
import numpy as np

# Load the trained model
with open('StudentsPerformance.pkl', 'rb') as f:
    model = pickle.load(f)

st.title("Student Performance Prediction")

# Input fields for the user
math_score = st.number_input("Math Score", min_value=0, max_value=100, value=50)
reading_score = st.number_input("Reading Score", min_value=0, max_value=100, value=50)

# Prediction button
if st.button("Predict Writing Score"):
    # Prepare input data
    input_data = np.array([[math_score, reading_score]])
    
    # Predict using the loaded model
    prediction = model.predict(input_data)
    
    st.write(f"Predicted Writing Score: {prediction[0]:.2f}")
