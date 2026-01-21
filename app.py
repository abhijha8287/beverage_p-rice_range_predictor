import streamlit as st
import pandas as pd
import joblib

# Load model and columns
model = joblib.load("best_model.pkl")
model_cols = joblib.load("model_columns.pkl")

st.set_page_config(page_title="Beverage Price Predictor", layout="centered")

st.title("🥤 Beverage Price Range Predictor")
st.write("Fill the details below to predict the price range")

# ==============================
# INPUT FORM
# ==============================
zone = st.selectbox("Zone", ["Urban", "Metro", "Semi-Urban", "Rural"])
occupation = st.selectbox("Occupation", ["Student", "Working Professional", "Entrepreneur", "Retired"])
income = st.selectbox("Income Level", ["<10L", "10L - 15L", "16L - 25L", "26L - 35L", "> 35L", "Not Reported"])
consume_freq = st.selectbox("Consume Frequency", ["0-2 times", "3-4 times", "5-7 times"])
aware = st.selectbox("Awareness of Other Brands", ["0 to 1", "2 to 4", "above 4"])
current_brand = st.selectbox("Current Brand", ["Established", "Newcomer"])
reason = st.selectbox("Reason for Choosing Brand", ["Price", "Quality", "Taste", "Availability", "Brand Image"])
health = st.selectbox("Health Concern", ["Low", "Medium", "High"])
packaging = st.selectbox("Packaging Preference", ["Simple", "Premium"])
channel = st.selectbox("Purchase Channel", ["Online", "Retail Store", "Supermarket"])
flavor = st.selectbox("Flavor Preference", ["Traditional", "Exotic", "Mixed"])
situation = st.selectbox("Consumption Situation", ["Active (eg. Sports, gym)", "Casual", "Party", "Travel"])
size = st.selectbox("Preferred Size", ["Small (250 ml)", "Medium (500 ml)", "Large (1L)"])
age_group = st.selectbox("Age Group", ["18-25","26-35","36-45","46-55","56-70","70+"])

# ==============================
# CREATE INPUT DATAFRAME
# ==============================
input_dict = {
    "zone": zone,
    "occupation": occupation,
    "income_levels": income,
    "consume_frequency(weekly)": consume_freq,
    "awareness_of_other_brands": aware,
    "current_brand": current_brand,
    "reasons_for_choosing_brands": reason,
    "health_concerns": health,
    "packaging_preference": packaging,
    "purchase_channel": channel,
    "flavor_preference": flavor,
    "typical_consumption_situations": situation,
    "preferable_consumption_size": size,
    "age_group": age_group
}

input_df = pd.DataFrame([input_dict])

# ==============================
# FEATURE ENGINEERING (SAME AS TRAINING)
# ==============================

# cf_ab_score
freq_map = {"0-2 times":1, "3-4 times":2, "5-7 times":3}
aware_map = {"0 to 1":1, "2 to 4":2, "above 4":3}

f = freq_map[consume_freq]
a = aware_map[aware]
input_df["cf_ab_score"] = round(f / (f + a), 2)

# zas_score
zone_map = {"Urban":3, "Metro":4, "Rural":1, "Semi-Urban":2}
income_map = {"<10L":1, "10L - 15L":2, "16L - 25L":3, "26L - 35L":4, "> 35L":5, "Not Reported":0}

z = zone_map[zone]
i = income_map[income]
input_df["zas_score"] = round(z / (z + i), 2)

# bsi
input_df["bsi"] = int((current_brand != "Established") and (reason in ["Price","Quality"]))

# ==============================
# ENCODING (One hot like training)
# ==============================
input_df = pd.get_dummies(input_df)

# Add missing columns
for col in model_cols:
    if col not in input_df.columns:
        input_df[col] = 0

# Reorder columns
input_df = input_df[model_cols]

# ==============================
# PREDICT
# ==============================
if st.button("Predict Price Range"):
    pred = model.predict(input_df)[0]

    mapping = {0:"100-150", 1:"150-200", 2:"200-300", 3:"300+"}
    st.success(f"💰 Predicted Price Range: **{mapping.get(pred, pred)}**")
