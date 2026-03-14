import streamlit as st
import numpy as np
import joblib

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Insurance Claim Prediction System")

age = st.slider("Age",18,70,30)

gender = st.selectbox("Gender",["Male","Female"])

policy = st.selectbox("Policy Type",["Basic","Premium","Gold"])

claim_amount = st.number_input("Claim Amount",1000,60000)

incident = st.selectbox("Incident Type",["Accident","Theft","Fire","Natural Disaster"])

gender_map = {"Male":0,"Female":1}
policy_map = {"Basic":0,"Premium":1,"Gold":2}
incident_map = {"Accident":0,"Theft":1,"Fire":2,"Natural Disaster":3}

input_data = np.array([[

age,
gender_map[gender],
policy_map[policy],
claim_amount,
incident_map[incident]

]])

if st.button("Predict"):

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)

    if prediction[0] == 1:
        st.success("Claim Approved")
    else:
        st.error("Claim Rejected")