import streamlit as st
import pickle
import numpy as np
import pandas as pd

# ================================
# Load the saved model
# ================================
with open("calories_best_model.pkl", "rb") as file:
    model = pickle.load(file)

# ================================
# Streamlit UI
# ================================
st.set_page_config(page_title="Calories Burnt Predictor", page_icon="🔥", layout="centered")

st.title("🔥 Calories Burnt Prediction App")
st.write("Enter your details below to estimate calories burnt during exercise.")

# ================================
# Input Fields
# ================================
gender = st.selectbox("Gender", ("Male", "Female"))
age = st.number_input("Age", min_value=10, max_value=100, value=25)
height = st.number_input("Height (cm)", min_value=100, max_value=250, value=170)
weight = st.number_input("Weight (kg)", min_value=30, max_value=200, value=70)
duration = st.number_input("Duration (minutes)", min_value=1, max_value=120, value=30)
heart_rate = st.number_input("Heart Rate (bpm)", min_value=50, max_value=200, value=100)
body_temp = st.number_input("Body Temperature (°C)", min_value=36.0, max_value=42.0, value=39.0)

# Convert Gender to numeric
gender_num = 0 if gender == "Male" else 1

# ================================
# Prediction
# ================================
if st.button("Predict Calories Burnt"):
    # Arrange input into correct shape
    input_data = np.array([[gender_num, age, height, weight, duration, heart_rate, body_temp]])
    prediction = model.predict(input_data)

    st.success(f"🔥 Estimated Calories Burnt: **{prediction[0]:.2f} kcal**")

# ================================
# Footer
# ================================
st.write("---")
st.caption("Built with ❤️ using Streamlit & Machine Learning")
