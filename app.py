import streamlit as st
import numpy as np
import joblib
import base64

# --- Background Styling with Translucent Card ---
st.markdown("""
<style>
.stApp {
    background-image: url('background.png');  /* Local image */
    background-size: cover;
    background-repeat: no-repeat;
    background-attachment: fixed;
}

/* Light translucent background for the main content area */
[data-testid="stAppViewContainer"] > .main {
    background-color: rgba(255, 255, 255, 0.85);
    padding: 2rem;
    border-radius: 15px;
    box-shadow: 0 0 10px rgba(0,0,0,0.3);
    max-width: 900px;
    margin: auto;
}

h1 {
    color: #222;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# --- Load the Extra Trees Model ---
model = joblib.load("extra_trees_model.pkl")

# --- App Title and Instructions ---
st.title("☀️ Solar Power Generation Predictor (Extra Trees Model)")
st.markdown("Fill in the environmental parameters below to estimate power generation:")

# --- Input Fields ---
distance_to_solar_noon = st.slider("Distance to Solar Noon", 0.0, 1.0, 0.5)
temperature = st.slider("Temperature (°F)", 0, 120, 75)
wind_direction = st.slider("Wind Direction (°)", 0, 360, 180)
wind_speed = st.slider("Wind Speed (mph)", 0.0, 50.0, 10.0)
sky_cover = st.slider("Sky Cover (%)", 0, 100, 25)
visibility = st.slider("Visibility (miles)", 0.0, 10.0, 10.0)
humidity = st.slider("Humidity (%)", 0, 100, 60)
avg_wind_speed = st.slider("Avg Wind Speed (mph)", 0.0, 50.0, 5.0)
avg_pressure = st.slider("Avg Pressure", 20.0, 35.0, 29.9)

# --- Derived Feature ---
solar_proximity = 1 - distance_to_solar_noon

# --- Prepare Feature Array ---
features = np.array([[distance_to_solar_noon, temperature, wind_direction,
                      wind_speed, sky_cover, visibility, humidity,
                      avg_wind_speed, avg_pressure, solar_proximity]])

# --- Prediction ---
if st.button("🔮 Predict Power"):
    prediction = model.predict(features)[0]
    st.success(f"Estimated Power Generated: **{int(prediction)} joules**")

# --- Footer ---
st.markdown("---")
st.markdown("📘 *Project by Lokesh Kumar Gadhi | July 2025*")