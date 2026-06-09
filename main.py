import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt

#To use, please use streamlit run main.py or py -m streamlit run main.py
model = joblib.load("rf_model.pkl")

st.title("Mumbai PM2.5 Predictor")
col1,col2,col3 = st.columns(3)

with col1:
    pm10 = st.number_input("PM10")
    no = st.number_input("NO")
    no2 = st.number_input("NO2")
    nox = st.number_input("NOx")
    nh3 = st.number_input("NH3")
    so2 = st.number_input("SO2")
with col2:
    co = st.number_input("CO")
    ozone = st.number_input("Ozone")
    benzene = st.number_input("Benzene")
    toluene = st.number_input("Toluene")
    eth_benzene = st.number_input("Eth-Benzene")
    mp_xylene = st.number_input("MP-Xylene")
with col3:
    rh = st.number_input("Relative Humidity")
    ws = st.number_input("Wind Speed")
    wd = st.number_input("Wind Direction")
    bp = st.number_input("Barometric Pressure")
    xylene = st.number_input("Xylene")
    at = st.number_input("Air Temperature")

if st.button("Predict"):
    data = pd.DataFrame([[
        pm10, no, no2, nox, nh3, so2, co,
        ozone, benzene, toluene, eth_benzene,
        mp_xylene, rh, ws, wd, bp, xylene, at
    ]], columns=[
        "PM10","NO","NO2","NOx","NH3","SO2","CO",
        "Ozone","Benzene","Toluene","Eth-Benzene",
        "MP-Xylene","RH","WS","WD","BP","Xylene","AT"
    ])

    prediction = model.predict(data)[0]

    st.success(f"Predicted PM2.5: {prediction:.2f}")

    if prediction <= 50:
        category = "Good"
    elif prediction <= 100:
        category = "Moderate"
    elif prediction <= 200:
        category = "Poor"
    else:
        category = "Very Poor"

    st.write("Category:", category)

    importance = pd.Series(
        model.feature_importances_,
        index=data.columns
    ).sort_values(ascending=False)

    fig, ax = plt.subplots()
    importance.head(10).plot(kind="bar", ax=ax)

    st.pyplot(fig)