# Telco Customer Churn Prediction

An end-to-end data science project that explores customer churn patterns and predicts churn using Logistic Regression, Random Forest, and XGBoost.

## Dataset
The dataset is named **`Telco_Customer_Churn.csv`** in this project. It contains 7,043 customer records and 21 columns, including the `Churn` target. `TotalCharges` is parsed as numeric and missing values are imputed within the model pipeline. `customerID` is excluded from model training.

## Project files
- `Customer_Churn_Prediction.ipynb` — complete analysis, EDA, model training, tuning, evaluation, feature importance, and export.
- `Telco_Customer_Churn.csv` — dataset supplied for this project.
- `app.py` — Streamlit dashboard with dataset insights, recommendations, model training, and a new-customer prediction form.
- `best_churn_model.joblib` — pre-trained best model exported from the notebook.
- `model_comparison.csv` and `feature_importance.csv` — evaluation and interpretation outputs.
- `requirements.txt` — Python dependencies.
- `DEPLOYMENT.md` — local run, GitHub, and Streamlit deployment instructions.

## Run locally
Use Python 3.10 or newer.

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

Run the notebook from this folder:

```bash
jupyter notebook
```

Run the dashboard:

```bash
streamlit run app.py
```

The app automatically loads `Telco_Customer_Churn.csv` and the included `best_churn_model.joblib` from the same folder. You can use the saved model immediately or train a different model from the sidebar. Alternatively, upload the CSV using the sidebar.

## Streamlit dashboard features
1. Dataset overview and data-quality preview.
2. Churn-rate charts by contract, tenure, payment method, and monthly charges.
3. Automatically generated segment insights.
4. Retention recommendations based on observed segment churn rates.
5. Train Logistic Regression, Random Forest, or XGBoost.
6. Enter a new customer's details and obtain an estimated churn probability.

## Deploy
Push this folder to GitHub, then deploy `app.py` using [Streamlit Community Cloud](https://share.streamlit.io/). Detailed steps are in `DEPLOYMENT.md`.

## Evaluation and responsible use
- Data is split into training, validation, and test partitions using stratification.
- Preprocessing is fitted inside model pipelines.
- Hyperparameters are tuned with stratified cross-validation on training data.
- Decision thresholds are selected on validation data.
- The test partition is held out for final evaluation.
- Churn segment patterns and feature importance are associations, not proof of causation.
- The app is an educational portfolio project, not a production customer decision system.
