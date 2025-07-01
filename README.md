# Telco Customer Churn Prediction

This project predicts customer churn in the telecom industry using machine learning, with exploratory data analysis (EDA), model evaluation, and interactive visualizations.

## Project Overview
- **Dataset**: [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) from Kaggle.
- **Objective**: Predict whether a customer will churn based on features like tenure, contract type, and monthly charges.
- **Tools Used**:
  - **Jupyter Lab**: For coding, EDA, and model development.
  - **Python Libraries**: Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Pickle, Streamlit.
  - **Power BI**: For visualization and classification result comparison.
  - **Streamlit**: For deploying an interactive web app.

## Repository Structure
- `EDA-TelcomChurn.ipynb & ML-TelcomChurn.ipynb`: Jupyter notebook with EDA, model training, and evaluation.
- `requirements.txt`: Python dependencies.
- `Compare_models.py`: Streamlit app for model deployment.
- `*.pkl`: Pickled machine learning models.
- `Telco-Curn-Visualization.pbix`: Power BI file for visualizations.
- `feature_importance.csv, Confustion_Matrix.vsc, metric_bestModel.csv, ML_prediction.csv, ROC_REsult.csv`: metrics and results of classification models
- `README.md`: Project documentation.

## Exploratory Data Analysis (EDA)
- Conducted in Jupyter Lab using Pandas, Matplotlib, and Seaborn.
- **Key Analyses**:
  - Distribution of churn vs. non-churn customers.
  - Correlation between features (e.g., tenure, monthly charges) and churn.
  - Visualizations: Bar plots, histograms, heatmaps, and pair plot.
- **Insights**:
  - Customers with shorter tenure are more likely to churn.
  - Higher monthly charges correlate with increased churn probability.
  - Contract type (month-to-month vs. long-term) significantly impacts churn.

## Machine Learning Models
- **Models Implemented** (using Scikit-learn):
  - Logistic Regression
  - Random Forest
  - Decision Tree
  - Support Vector Machine (SVM)
- **Evaluation**:
  - Metrics: Accuracy, precision, recall, F1-score, and ROC-AUC.
  - Visualizations: feature importance plots.
- **Key Findings**:
  - Random Forest outperformed others in accuracy and F1-score.
  - Logistic Regression provided interpretable coefficients for feature impact.
  - SVM showed robust performance but was computationally intensive.
  - Decision Tree was prone to overfitting but useful for feature insights.

## Model Deployment
- **Pickle**: Trained model saved as `*.pkl` for reuse.
- **Streamlit**: Interactive web app (`Compare_models.py`) allows users to input customer data and predict churn probability.
- **Run the App**:

  ```bash
  pip install -r requirements.txt
  python streamlit run Compare_models.py
