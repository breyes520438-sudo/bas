import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load the trained model
model = joblib.load('logistic_regression_model.pkl')

st.title('Red Wine Quality Predictor')

st.write("""
Enter the chemical attributes of the red wine to predict if it is of 'Good Quality' (quality >= 7).
""")

# Define input fields for the 11 features
fixed_acidity = st.number_input('Fixed Acidity', value=7.4, format="%.4f")
volatile_acidity = st.number_input('Volatile Acidity', value=0.70, format="%.4f")
citric_acid = st.number_input('Citric Acid', value=0.00, format="%.4f")
residual_sugar = st.number_input('Residual Sugar', value=1.9, format="%.4f")
chlorides = st.number_input('Chlorides', value=0.076, format="%.4f")
free_sulfur_dioxide = st.number_input('Free Sulfur Dioxide', value=11.0, format="%.4f")
total_sulfur_dioxide = st.number_input('Total Sulfur Dioxide', value=34.0, format="%.4f")
density = st.number_input('Density', value=0.9978, format="%.5f")
pH = st.number_input('pH', value=3.51, format="%.4f")
sulphates = st.number_input('Sulphates', value=0.56, format="%.4f")
alcohol = st.number_input('Alcohol', value=9.4, format="%.4f")

# Create a button to trigger prediction
if st.button('Predict Quality'):
    # Collect input data into a DataFrame
    input_data = pd.DataFrame([[
        fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
        chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density,
        pH, sulphates, alcohol
    ]], columns=[
        'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
        'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
        'pH', 'sulphates', 'alcohol'
    ])

    # Make prediction
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)

    # Display the result
    if prediction[0] == 1:
        st.success('Prediction: Good Quality Wine')
        confidence = prediction_proba[0][1] * 100
        st.write(f'Confidence: {confidence:.2f}%')
    else:
        st.error('Prediction: Not Good Quality Wine')
        confidence = prediction_proba[0][0] * 100
        st.write(f'Confidence: {confidence:.2f}%')
