{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "22d04a93",
   "metadata": {},
   "source": [
    "# Telco Customer Churn Prediction\n",
    "\n",
    "This project predicts customer churn in the telecom industry using machine learning, with exploratory data analysis (EDA), model evaluation, and interactive visualizations.\n",
    "\n",
    "## Project Overview\n",
    "- **Dataset**: [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) from Kaggle.\n",
    "- **Objective**: Predict whether a customer will churn based on features like tenure, contract type, and monthly charges.\n",
    "- **Tools Used**:\n",
    "  - **Jupyter Lab**: For coding, EDA, and model development.\n",
    "  - **Python Libraries**: Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Pickle, Streamlit.\n",
    "  - **Power BI**: For visualization and classification result comparison.\n",
    "  - **Streamlit**: For deploying an interactive web app.\n",
    "\n",
    "## Repository Structure\n",
    "- `EDA-TelcomChurn.ipynb & ML-TelcomChurn.ipynb`: Jupyter notebook with EDA, model training, and evaluation.\n",
    "- `requirements.txt`: Python dependencies.\n",
    "- `Compare_models.py`: Streamlit app for model deployment.\n",
    "- `*.pkl`: Pickled machine learning models.\n",
    "- `Telco-Curn-Visualization.pbix`: Power BI file for visualizations.\n",
    "- `feature_importance.csv, Confustion_Matrix.vsc, metric_bestModel.csv, ML_prediction.csv, ROC_REsult.csv`: metrics and results of classification models\n",
    "- `README.md`: Project documentation.\n",
    "\n",
    "## Exploratory Data Analysis (EDA)\n",
    "- Conducted in Jupyter Lab using Pandas, Matplotlib, and Seaborn.\n",
    "- **Key Analyses**:\n",
    "  - Distribution of churn vs. non-churn customers.\n",
    "  - Correlation between features (e.g., tenure, monthly charges) and churn.\n",
    "  - Visualizations: Bar plots, histograms, heatmaps, and pair plot.\n",
    "- **Insights**:\n",
    "  - Customers with shorter tenure are more likely to churn.\n",
    "  - Higher monthly charges correlate with increased churn probability.\n",
    "  - Contract type (month-to-month vs. long-term) significantly impacts churn.\n",
    "\n",
    "## Machine Learning Models\n",
    "- **Models Implemented** (using Scikit-learn):\n",
    "  - Logistic Regression\n",
    "  - Random Forest\n",
    "  - Decision Tree\n",
    "  - Support Vector Machine (SVM)\n",
    "- **Evaluation**:\n",
    "  - Metrics: Accuracy, precision, recall, F1-score, and ROC-AUC.\n",
    "  - Visualizations: feature importance plots.\n",
    "- **Key Findings**:\n",
    "  - Random Forest outperformed others in accuracy and F1-score.\n",
    "  - Logistic Regression provided interpretable coefficients for feature impact.\n",
    "  - SVM showed robust performance but was computationally intensive.\n",
    "  - Decision Tree was prone to overfitting but useful for feature insights.\n",
    "\n",
    "## Model Deployment\n",
    "- **Pickle**: Trained model saved as `*.pkl` for reuse.\n",
    "- **Streamlit**: Interactive web app (`Compare_models.py`) allows users to input customer data and predict churn probability.\n",
    "- **Run the App**:\n",
    "\n",
    "  ```bash\n",
    "  \n",
    "  `pip install -r requirements.txt`\n",
    "  `python streamlit run Compare_models.py` if you install streamlit as --user, run the following command: `python -m streamlit run Compare_models.py`"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.9"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
