# housingapp.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf

# --- Load artifacts ---
model = tf.keras.models.load_model("artifacts/housing_model.h5")
scaler = joblib.load("artifacts/scaler.pkl")
features = joblib.load("artifacts/feature_names.pkl")

# --- Streamlit UI ---
st.title("🏠 Hamilton Housing Appraiser")

st.write(
    "Enter property details to predict the appraised value in Hamilton."
)

# Input widgets
calc_acres = st.number_input("Lot Size (Acres)", min_value=0.0, step=0.01, value=0.5)

# Get unique options from saved feature names
land_use_options = [f.replace("LAND_USE_CODE_DESC_", "") 
                    for f in features if f.startswith("LAND_USE_CODE_DESC_")]
land_use_options = ["Unknown"] + land_use_options
property_type_options = [f.replace("PROPERTY_TYPE_CODE_DESC_", "") 
                         for f in features if f.startswith("PROPERTY_TYPE_CODE_DESC_")]
property_type_options = ["Unknown"] + property_type_options

land_use = st.selectbox("Land Use", land_use_options)
property_type = st.selectbox("Property Type", property_type_options)

# --- Prepare input for prediction ---
def preprocess_input(calc_acres, land_use, property_type):
    # Initialize dataframe with zeros
    input_dict = {feat: 0 for feat in features}
    input_dict["CALC_ACRES"] = calc_acres
    
    # Set categorical flags
    land_feat = f"LAND_USE_CODE_DESC_{land_use}"
    prop_feat = f"PROPERTY_TYPE_CODE_DESC_{property_type}"
    
    if land_feat in input_dict:
        input_dict[land_feat] = 1
    if prop_feat in input_dict:
        input_dict[prop_feat] = 1
    
    # Convert to dataframe
    df_input = pd.DataFrame([input_dict])
    
    # Scale numeric
    df_input_scaled = scaler.transform(df_input)
    return df_input_scaled

# --- Prediction ---
if st.button("Predict Appraised Value"):
    X_input = preprocess_input(calc_acres, land_use, property_type)
    prediction = model.predict(X_input)[0][0]
    st.success(f"Estimated Appraised Value: ${prediction:,.2f}")
