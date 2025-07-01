import streamlit as st
import pandas as pd
import pickle

pipeline_dic = {
    'LogisticRegression': 'Logistic_churn_prediction_pipeline.pkl',
    'DecisonTree': 'Decision Tree_churn_prediction_pipeline.pkl',
    'RandomForest' : 'Random Forest_churn_prediction_pipeline.pkl'
}
 

st.title("Prediction Customer Churn (file & manual)")

st.markdown("**Objective:** Analyze the Telco Customer Churn dataset to predict customer churn using supervised (Logistic Regression, Decision Tree, Random Forest) methods.")
st.markdown("**Dataset:**")
st.markdown("""
- **Source**: [Kaggle Telco Customer Churn Dataset] (http://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Features**: Customer demographics (age, gender), services (internet, phone, streaming, support), contract details, payment methods, monthly charges, tenure, etc.
- **Target**: Churn (Yes/No)
""")

#Separate in two column
input_col, output_col = st.columns([1, 2])

with input_col: 
    st.subheader("All input")
    #Upload file
    uploaded_file = st.file_uploader("Choose/Drag csv file", type = ["csv"])

    pipeline = {}
    # Load the pipeline
    for name, pipefile in pipeline_dic.items():
        with open(pipefile, 'rb') as file:
            pipeline[name] = pickle.load(file)

with output_col:
    st.subheader("All output")
    #Read csv file
    if uploaded_file is not None:
        dataset = pd.read_csv(uploaded_file)
        st.write("Data overview: ", dataset.head())

    # Make prediction
        for name in pipeline:
            dataset[f"{name}_predict"] = pipeline[name].predict(dataset)

        st.markdown("### Overal summary")

        summary_df = pd.DataFrame(index=["Loyal", "Churn"])
        total = len(dataset)
        for name in pipeline:
            preds = dataset[f"{name}_predict"]
            loyal = (preds == 0).sum()
            churn = (preds == 1).sum()
            churn_pct = churn / total * 100
            loyal_pct = loyal / total * 100
            summary_df[name]  = [f"{loyal} ({loyal_pct:.1f}%)", f"{churn} ({churn_pct:.1f}%)"]

        summary_df.index.name = "Class"
        st.dataframe(summary_df)

        st.markdown("### Overall result of models")
        st.dataframe(dataset.head(10))

        #Download file
        csv_result = dataset.to_csv(index=False).encode('utf-8')
        st.download_button(label="Download file with prediction", data=csv_result, file_name="Churn_prediction.csv", mime="text/csv")


with input_col: 
    with st.form("Input customer"):
        st.markdown("Insert customer caracteristics manually")
        left, right = st.columns([1, 2])
        #Fields form
        with left: 
            gender = st.selectbox("Gender", ["Male", "Female"])
            senior = st.selectbox("Senior Citizen", [0, 1])
            partner = st.selectbox("Partner ?", ["Yes", "No"])
            dependents = st.selectbox("Dependents ?", ["Yes", "No"])
            tenure = st.slider("Tenure", 0, 72, 12)
            phone_service = st.selectbox("Phone service ?", ["Yes", "No"])
            multiple_lines = st.selectbox("Multiple line ?", ["Yes", "No", "No phone service"])
            internet_service = st.selectbox("Internet setvice", ["DSL", "Fiber optic", "No"])
            online_security = st.selectbox("Online security ?", ["Yes", "No", "No internet service"])
        with right:
            online_backup = st.selectbox("Online back up ?", ["Yes", "No", "No internet service"])
            device_protection = st.selectbox("Device protection ?", ["Yes", "No", "No internet service"])
            tech_support = st.selectbox("Technic support ?", ["Yes", "No", "No internet service"])
            streaming_tv = st.selectbox("Streaming TV ?", ["Yes", "No", "No internet service"])
            streaming_movies = st.selectbox("Streaming Movies ?", ["Yes", "No", "No internet service"])
            contract = st.selectbox("Contract type ?", ["Month-to-month", "One year", "Two year"])
            paperless_billing = st.selectbox("Paperless billing ?", ["Yes", "No"])
            payment_method = st.selectbox("Payement method ?", [
                "Electronic check", "Mailed check",
                "Bank transfer (automatic)", "Credit card (automatic)"])
            monthly_charges = st.number_input("Monthly charges", 0.0, 200.0, 70.0)
            total_charges = st.number_input("Total charges", 0.0, 10000.0, 250.0)

        submited = st.form_submit_button("Submit")

with output_col:
        # Bouton prédiction
        if submited:
            input_data = pd.DataFrame([{
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
            }])

            #Result
            st.markdown("### Result of manual input:")
            col1, col2, col3 = st.columns(3) 
            for col, name in zip([col1, col2, col3], pipeline.keys()):
                try:
                    prediction = pipeline[name].predict(input_data)[0]
                    proba = pipeline[name].predict_proba(input_data)[0][1]
                    msg = "Churn" if prediction == 1 else "Loyal"
                    col.metric(label=name, value=msg, delta=f"{proba*100:.1f}")
                except Exception as e:
                    col.error(f"{name} : error - {e}")
