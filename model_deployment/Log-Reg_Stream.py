import streamlit as st
import pandas as pd
import pickle


# Load the pipeline
with open('Logistic_churn_prediction_pipeline.pkl', 'rb') as file:
    pipeline = pickle.load(file)

    
#Upload file
uploaded_file = st.file_uploader("Choose csv file", type = ["csv"])

#Read csv file
if uploaded_file is not None:
    dataset = pd.read_csv(uploaded_file)
    st.write("Data overview: ", dataset.head())

# Convert to DataFrame
#new_data = pd.DataFrame([data])
#    dataset = dataset.reindex(columns=all_cols, fill_value=pd.NA)

# Make prediction
    #prediction = pipeline.predict(dataset)[0]
    prediction = pipeline.predict(dataset)
    dataset['Prediction'] = prediction

    # Affichage des résultats
    st.subheader("📊 Prediction Resume:")

    total = len(dataset)
    churned = dataset['Prediction'].sum()
    loyal = total - churned
    churn_pct = churned / total * 100
    loyal_pct = loyal / total * 100

    st.write(f"- Customers number : **{total}**")
    st.success(f"- Loyal customers : **{loyal}** ({loyal_pct:.1f}%)")
    st.error(f"- Churners customers : **{churned}** ({churn_pct:.1f}%)")

    # Statistiques supplémentaires
    st.write("---")
    st.subheader("📈 Supplementary stat:")
    st.write(f"Monthly means : {dataset['MonthlyCharges'].mean():.2f} $")
    st.write(f"Tenure means : {dataset['tenure'].mean():.1f} mois")

    # Distribution des contrats
    if 'Contract' in dataset.columns:
        st.write("Contract type :")
        st.dataframe(dataset.groupby('Contract')['Prediction'].value_counts())

    # Aperçu avec prédictions
    st.subheader("🧾 Data with prediction")

    st.write(dataset.head())

    #Download file
    csv_result = dataset.to_csv(index=False).encode('utf-8')
    st.download_button(label="Download file with prediction", data=csv_result, file_name="Churn_prediction.csv", mime="text/csv")


# --- Partie 2 : FORMULAIRE MANUEL ---
st.header("🧾 Manual prediction")

# Champs du formulaire (adaptés au dataset Telco)
gender = st.selectbox("Gender", ["Male", "Female"])
senior = st.selectbox("Senior Citizen", [0, 1])
partner = st.selectbox("Partner ?", ["Yes", "No"])
dependents = st.selectbox("Dependents ?", ["Yes", "No"])
tenure = st.slider("Tenure", 0, 72, 12)
phone_service = st.selectbox("Phone service ?", ["Yes", "No"])
multiple_lines = st.selectbox("Multiple line ?", ["Yes", "No", "No phone service"])
internet_service = st.selectbox("internet setvice", ["DSL", "Fiber optic", "No"])
online_security = st.selectbox("Online security ?", ["Yes", "No", "No internet service"])
online_backup = st.selectbox("Online back up ?", ["Yes", "No", "No internet service"])
device_protection = st.selectbox("Device protection ?", ["Yes", "No", "No internet service"])
tech_support = st.selectbox("Technic support ?", ["Yes", "No", "No internet service"])
streaming_tv = st.selectbox("Streaming TV ?", ["Yes", "No", "No internet service"])
streaming_movies = st.selectbox("Streaming Movies ?", ["Yes", "No", "No internet service"])
contract = st.selectbox("Cpntract type", ["Month-to-month", "One year", "Two year"])
paperless_billing = st.selectbox("Paperless billing ?", ["Yes", "No"])
payment_method = st.selectbox("Payement method", [
    "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
])
monthly_charges = st.number_input("Monthly charges", 0.0, 200.0, 70.0)
total_charges = st.number_input("Total charges", 0.0, 10000.0, 250.0)

# Bouton prédiction
if st.button("📌 Predict"):
    input_data = pd.DataFrame{
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges
    }

    single_dataset = pd.DataFrame([input_dict])

    # Prediction
    prediction = pipeline.predict(single_dataset)[0]
    prob = pipeline.predict_proba(single_dataset)[0][1]

    # Affichage
    st.markdown("### Result :")
    if prediction == 1:
        st.error(f"🔴 Churner risk ({prob:.2%})")
    else:
        st.success(f"🟢 Loyal customer ({(1-prob):.2%})")