import pickle
import numpy as np
import pandas as pd
import streamlit as st

# ============================
# Load trained model & encoder
# ============================
with open("xgb_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("xgb_model.pkl", "rb") as f:
    encoder = pickle.load(f)   # assume you saved LabelEncoder/OneHotEncoder

st.title("💳 Bank Account Ownership Prediction")
st.write("Enter details to predict whether a respondent has a bank account.")

# ============================
# User Input Form
# ============================

country = st.selectbox("Country", ["Kenya", "Rwanda", "Tanzania", "Uganda"])
year = st.selectbox("Year", [2016, 2017, 2018])
location_type = st.selectbox("Location Type", ["Urban", "Rural"])
cellphone_access = st.selectbox("Cellphone Access", ["Yes", "No"])
household_size = st.number_input("Household Size", min_value=1, max_value=50, value=3)
age = st.number_input("Age of Respondent", min_value=16, max_value=100, value=30)
gender = st.selectbox("Gender", ["Male", "Female"])
relationship = st.selectbox(
    "Relationship with Head",
    ["Head of Household", "Spouse", "Child", "Parent", "Other relative", "Other non-relatives"],
)
marital_status = st.selectbox(
    "Marital Status",
    ["single", "married", "widowed", "divorced"]
)
education_level = st.selectbox(
    "Education Level",
    ["none", "primary", "secondary", "vocational", "tertiary"]
)
job_type = st.selectbox(
    "Job Type",
    ["Self employed", "Government Dependent", "Formally employed Government",
     "Formally employed Private", "Informally employed", "Other Income", "No Income"]
)

# ============================
# Create DataFrame
# ============================

input_dict = {
    "country": country,
    "year": year,
    "location_type": location_type,
    "cellphone_access": cellphone_access,
    "household_size": household_size,
    "age_of_respondent": age,
    "gender_of_respondent": gender,
    "relationship_with_head": relationship,
    "marital_status": marital_status,
    "education_level": education_level,
    "job_type": job_type,
}

input_df = pd.DataFrame([input_dict])

# ============================
# Apply same preprocessing as training
# ============================

# If you used encoder (OneHotEncoder / LabelEncoder), transform here:
try:
    input_encoded = encoder.transform(input_df)
except:
    # if model can handle raw categorical (like CatBoost/XGBoost with `enable_categorical`), just pass DataFrame
    input_encoded = input_df

# ============================
# Prediction
# ============================

if st.button("🔮 Predict"):
    prediction = model.predict(input_encoded)[0]
    probability = model.predict_proba(input_encoded)[0]

    st.subheader("✅ Prediction Result")
    st.write(f"**Predicted Bank Account Ownership:** {'Yes' if prediction == 1 else 'No'}")
    st.write("📊 Class Probabilities:")
    st.write({"No": probability[0], "Yes": probability[1]})
