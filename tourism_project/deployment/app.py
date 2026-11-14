import os
import streamlit as st
import pandas as pd
import joblib

# ==========================================
# 1. Model Loading
# ==========================================
# Resolve the model path relative to this script so it works reliably on Streamlit Community Cloud.
# Ensure the filename matches the exact artifact name saved and committed by train.py.
model_path = os.path.join(os.path.dirname(__file__), "best_tourism_model_v1.joblib")
model = joblib.load(model_path)

# ==========================================
# 2. Application Header & UI Inputs
# ==========================================
st.title("Tourism Package Prediction")
st.write("Enter the customer's demographic and interaction details below to predict package purchase likelihood.")

# Collect demographic and sales interaction inputs from the user
Age = st.slider("Age", min_value=18, max_value=70, value=30)
TypeofContact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
CityTier = st.selectbox("City Tier", [1, 2, 3])
DurationOfPitch = st.slider("Duration of Pitch (mins)", min_value=0, max_value=100, value=15)
Occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
Gender = st.selectbox("Gender", ["Male", "Female", "Others"])
NumberOfPersonVisiting = st.slider("Number of Persons Visiting", min_value=1, max_value=5, value=2)
NumberOfFollowups = st.slider("Number of Follow-ups", min_value=1, max_value=10, value=3)
ProductPitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
PreferredPropertyStar = st.selectbox("Preferred Property Star", [1, 2, 3, 4, 5])
MaritalStatus = st.selectbox("Marital Status", ["Married", "Single", "Divorced", "Unmarried"])
NumberOfTrips = st.slider("Number of Trips", min_value=1, max_value=20, value=3)
Passport = st.selectbox("Has Passport?", ["Yes", "No"])
PitchSatisfactionScore = st.slider("Pitch Satisfaction Score", min_value=1, max_value=5, value=3)
OwnCar = st.selectbox("Owns a Car?", ["Yes", "No"])
NumberOfChildrenVisiting = st.slider("Number of Children Visiting", min_value=0, max_value=5, value=1)
Designation = st.selectbox("Designation", ["Executive", "Manager", "AVP", "VP", "Sr. Manager"])
MonthlyIncome = st.number_input("Monthly Income", min_value=1000.0, value=30000.0)

# ==========================================
# 3. Feature Formatting
# ==========================================
# Assemble inputs into a single-row DataFrame matching the training schema.
# Categorical strings are passed raw so the pipeline's OneHotEncoder handles them identically to training.
#input_data = pd.DataFrame([{
#    'Age': Age,
#    'TypeofContact': TypeofContact,
#    'CityTier': CityTier,
#    'DurationOfPitch': DurationOfPitch,
#    'Occupation': Occupation,
#    'Gender': Gender,
#    'NumberOfPersonVisiting': NumberOfPersonVisiting,
#    'NumberOfFollowups': NumberOfFollowups,
#    'ProductPitched': ProductPitched,
#    'PreferredPropertyStar': PreferredPropertyStar,
#    'MaritalStatus': MaritalStatus,
#    'NumberOfTrips': NumberOfTrips,
#    'Passport': 1 if Passport == "Yes" else 0,
#    'PitchSatisfactionScore': PitchSatisfactionScore,
#    'OwnCar': 1 if OwnCar == "Yes" else 0,
#    'NumberOfChildrenVisiting': NumberOfChildrenVisiting,
#    'Designation': Designation,
#    'MonthlyIncome': MonthlyIncome
#}])

# Explicitly cast string/object columns to category dtype to satisfy XGBoost's enable_categorical=True
#categorical_cols = input_data.select_dtypes(include=['object', 'string']).columns
#for col in categorical_cols:
#    input_data[col] = input_data[col].astype('category')


# ===
# ==========================================
# 3. Feature Formatting & Alignment
# ==========================================
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

# Safely align with model features if present
if hasattr(model, "feature_names_in_"):
    # Add any missing model features as zeros/default
    for col in model.feature_names_in_:
        if col not in input_data.columns:
            input_data[col] = 0
    # Reindex strictly to model expectation
    input_data = input_data[model.feature_names_in_]

# Explicitly cast string/object columns to category dtype for XGBoost
categorical_cols = input_data.select_dtypes(include=['object', 'string']).columns
for col in categorical_cols:
    input_data[col] = input_data[col].astype('category')

# ==========================================
# 4. Inference & Prediction Display
# ==========================================
# Apply the tuned probability threshold to balance precision and recall
classification_threshold = 0.45

if st.button("Predict"):
    prob = model.predict_proba(input_data)[0, 1]
    pred = int(prob >= classification_threshold)
    result = "will purchase the travel package" if pred == 1 else "is unlikely to purchase"
    st.subheader(f"Prediction: Customer {result}")
    st.caption(f"Purchase Probability: {prob:.2%} (Threshold: {classification_threshold:.0%})")
