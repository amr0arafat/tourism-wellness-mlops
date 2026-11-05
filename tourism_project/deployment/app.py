"""Streamlit front-end for the Wellness Tourism Package purchase-prediction model."""
import os

import joblib
import pandas as pd
import streamlit as st

# The model is committed to the same folder by the GitHub Actions pipeline
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best_tourism_model_v1.joblib")
THRESHOLD = 0.45   # same decision threshold used during training


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


st.set_page_config(page_title="Wellness Package Purchase Predictor", page_icon="🧳", layout="wide")
st.title("🧳 Wellness Tourism Package – Purchase Prediction")
st.write("Enter the customer's details and pitch information to predict whether they "
         "are likely to purchase the **Wellness Tourism Package** before contacting them.")

if not os.path.exists(MODEL_PATH):
    st.error("Model file not found. Run the GitHub Actions pipeline so it commits the trained model.")
    st.stop()
model = load_model()

# ---------------------------------------------------------------- inputs
st.subheader("Customer details")
c1, c2, c3 = st.columns(3)
with c1:
    age = st.number_input("Age", min_value=18, max_value=100, value=35)
    gender = st.selectbox("Gender", ["Male", "Female"])
    marital_status = st.selectbox("Marital status", ["Married", "Single", "Unmarried", "Divorced"])
    occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
    designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
with c2:
    monthly_income = st.number_input("Monthly income", min_value=0.0, value=23000.0, step=500.0)
    city_tier = st.selectbox("City tier", [1, 2, 3])
    passport = st.selectbox("Has a valid passport?", ["Yes", "No"])
    own_car = st.selectbox("Owns a car?", ["Yes", "No"])
    number_of_trips = st.number_input("Average trips per year", min_value=0, max_value=30, value=3)
with c3:
    number_of_person_visiting = st.number_input("Number of people visiting", min_value=1, max_value=10, value=3)
    number_of_children_visiting = st.number_input("Children (<5 yrs) visiting", min_value=0, max_value=5, value=1)
    preferred_property_star = st.selectbox("Preferred hotel rating", [3, 4, 5])

st.subheader("Sales interaction")
c4, c5 = st.columns(2)
with c4:
    type_of_contact = st.selectbox("Type of contact", ["Self Enquiry", "Company Invited"])
    product_pitched = st.selectbox("Product pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
    duration_of_pitch = st.number_input("Duration of pitch (minutes)", min_value=0, max_value=200, value=15)
with c5:
    number_of_followups = st.number_input("Number of follow-ups", min_value=0, max_value=10, value=4)
    pitch_satisfaction_score = st.slider("Pitch satisfaction score", 1, 5, 3)

# Collect the inputs into a single-row dataframe with the training column names
input_df = pd.DataFrame([{
    "Age": age,
    "TypeofContact": type_of_contact,
    "CityTier": city_tier,
    "DurationOfPitch": duration_of_pitch,
    "Occupation": occupation,
    "Gender": gender,
    "NumberOfPersonVisiting": number_of_person_visiting,
    "NumberOfFollowups": number_of_followups,
    "ProductPitched": product_pitched,
    "PreferredPropertyStar": preferred_property_star,
    "MaritalStatus": marital_status,
    "NumberOfTrips": number_of_trips,
    "Passport": 1 if passport == "Yes" else 0,
    "PitchSatisfactionScore": pitch_satisfaction_score,
    "OwnCar": 1 if own_car == "Yes" else 0,
    "NumberOfChildrenVisiting": number_of_children_visiting,
    "Designation": designation,
    "MonthlyIncome": monthly_income,
}])

with st.expander("Show input data"):
    st.dataframe(input_df)

# ---------------------------------------------------------------- prediction
if st.button("Predict", type="primary"):
    probability = float(model.predict_proba(input_df)[0, 1])
    if probability >= THRESHOLD:
        st.success(f"✅ Likely to PURCHASE the Wellness Tourism Package (probability {probability:.1%}). "
                   "Prioritise this customer for outreach.")
    else:
        st.warning(f"❌ Unlikely to purchase (probability {probability:.1%}). "
                   "Lower priority for this campaign.")
    st.progress(min(max(probability, 0.0), 1.0))
