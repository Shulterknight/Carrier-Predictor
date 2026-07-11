import streamlit as st
import pandas as pd
import pickle

# Set up page config
st.set_page_config(page_title="Career Interest Predictor", layout="centered")

st.title(" Career Interest Predictor")
st.write("Select your skill levels below to find your recommended career path.")

# Load the saved model and label encoder
@st.cache_resource
def load_assets():
    with open('career_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('label_encoder.pkl', 'rb') as f:
        le = pickle.load(f)
    return model, le

try:
    rf_model, le = load_assets()
except FileNotFoundError:
    st.error("Model files not found. Ensure 'career_model.pkl' and 'label_encoder.pkl' are in the same folder.")
    st.stop()

# Define the expected features from your dataset structure
features = [
    'Database Fundamentals', 'Computer Architecture', 'Distributed Computing Systems',
    'Cyber Security', 'Networking', 'Software Development', 'Programming Skills',
    'Project Management', 'Computer Forensics Fundamentals', 'AI ML',
    'Software Engineering', 'Business Analysis', 'Data Science',
    'Troubleshooting skills', 'Graphics Designing'
]

# Mapping dictionary for skill levels
input_mapping = {
    'Not Interested': 0,
    'Poor': 1,
    'Beginner': 2,
    'Average': 3,
    'Intermediate': 4,
    'Professional': 5
}

# Render selection boxes for the user
user_data = {}
for col in features:
    choice = st.selectbox(
        f"Your level for {col}:",
        options=list(input_mapping.keys()),
        index=2 # Defaults to 'Beginner'
    )
    user_data[col] = [input_mapping[choice]]

# Predict button
if st.button("Predict My Career Path", type="primary"):
    input_df = pd.DataFrame(user_data)
    
    # Generate prediction
    predicted_numeric_id = rf_model.predict(input_df)[0]
    predicted_role = le.inverse_transform([predicted_numeric_id])[0]
    
    st.success(f"### 🎉 Recommended Career Path: **{predicted_role}**")
