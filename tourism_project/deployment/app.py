import streamlit as st
import pandas as pd
import joblib
import os

st.title("Tourism Package Purchase Prediction")

# Load model safely
@st.cache_resource
def load_model():
    model_path = "tourism_project/deployment/best_tourism_model_v1.joblib"
    return joblib.load(model_path)

model = load_model()

st.write("Please enter customer details below:")

# Input fields
Age = st.number_input("Age", min_value=18, max_value=100, value=35)
TypeofContact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
CityTier = st.selectbox("City Tier", [1, 2, 3])
DurationOfPitch = st.number_input("Duration of Pitch (minutes)", min_value=1, max_value=120, value=15)
Occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
Gender = st.selectbox("Gender", ["Male", "Female"])
NumberOfPersonVisiting = st.number_input("Number of Persons Visiting", min_value=1, max_value=10, value=2)
NumberOfFollowups = st.number_input("Number of Follow-ups", min_value=0, max_value=10, value=3)
ProductPitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
PreferredPropertyStar = st.selectbox("Preferred Property Star", [3, 4, 5])
MaritalStatus = st.selectbox("Marital Status", ["Single", "Married", "Unmarried", "Divorced"])
NumberOfTrips = st.number_input("Number of Trips per Year", min_value=0, max_value=20, value=2)
Passport = st.selectbox("Has Passport?", ["Yes", "No"])
PitchSatisfactionScore = st.slider("Pitch Satisfaction Score", 1, 5, 3)
OwnCar = st.selectbox("Owns Car?", ["Yes", "No"])
NumberOfChildrenVisiting = st.number_input("Number of Children Visiting (under 5)", min_value=0, max_value=5, value=0)
Designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
MonthlyIncome = st.number_input("Monthly Income", min_value=1000.0, max_value=1000000.0, value=20000.0)

if st.button("Predict"):
    input_data = pd.DataFrame([{
        'Age': Age,
        'TypeofContact': TypeofContact,
        'CityTier': CityTier,
        'DurationOfPitch': DurationOfPitch,
        'Occupation': Occupation,
        'Gender': Gender,
        'NumberOfPersonVisiting': NumberOfPersonVisiting,
        'NumberOfFollowups': NumberOfFollowups,
        'ProductPitched': ProductPitched,
        'PreferredPropertyStar': PreferredPropertyStar,
        'MaritalStatus': MaritalStatus,
        'NumberOfTrips': NumberOfTrips,
        'Passport': 1 if Passport == "Yes" else 0,
        'PitchSatisfactionScore': PitchSatisfactionScore,
        'OwnCar': 1 if OwnCar == "Yes" else 0,
        'NumberOfChildrenVisiting': NumberOfChildrenVisiting,
        'Designation': Designation,
        'MonthlyIncome': MonthlyIncome
    }])

    # Safely align columns with model expectation without raising KeyError
    if hasattr(model, "feature_names_in_"):
        for col in model.feature_names_in_:
            if col not in input_data.columns:
                input_data[col] = 0
        input_data = input_data[model.feature_names_in_]

    # Cast string/object columns to category for XGBoost
    categorical_cols = input_data.select_dtypes(include=['object', 'string']).columns
    for col in categorical_cols:
        input_data[col] = input_data[col].astype('category')

    try:
        prob = model.predict_proba(input_data)[0, 1]
        st.success(f"Prediction Complete! Purchase Probability: {prob:.2f}")
    except Exception as e:
        st.error(f"Prediction Error: {e}")
