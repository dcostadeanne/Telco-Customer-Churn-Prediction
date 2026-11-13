from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)
from xgboost import XGBClassifier

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET = BASE_DIR / "Telco_Customer_Churn.csv"
ARTIFACT_PATH = BASE_DIR / "best_churn_model.joblib"
RANDOM_STATE = 42

st.set_page_config(page_title="Telco Customer Churn", page_icon="📊", layout="wide")

st.title("📊 Telco Customer Churn Analytics")
st.markdown(
    "Explore customer churn patterns, understand the key risk segments, "
    "review retention recommendations, and predict churn risk for a new customer."
)

@st.cache_data
def load_dataset():
    if not DEFAULT_DATASET.exists():
        raise FileNotFoundError(
            "Telco_Customer_Churn.csv was not found beside app.py."
        )
    frame = pd.read_csv(DEFAULT_DATASET)
    if "TotalCharges" in frame.columns:
        frame["TotalCharges"] = pd.to_numeric(frame["TotalCharges"], errors="coerce")
    return frame

def validate_data(frame):
    if "Churn" not in frame.columns:
        raise ValueError("The dataset must contain a 'Churn' target column.")
    if not set(frame["Churn"].dropna().astype(str).unique()).issubset({"Yes", "No"}):
        raise ValueError("Churn values must be 'Yes' and 'No'.")
    return frame.dropna(subset=["Churn"]).drop_duplicates().copy()

def build_preprocessor(X):
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    prep = ColumnTransformer([
        ("num", numeric_pipe, numeric_cols),
        ("cat", categorical_pipe, categorical_cols)
    ])
    return prep, numeric_cols, categorical_cols

def create_model(model_name, X_train, y_train):
    prep, numeric_cols, categorical_cols = build_preprocessor(X_train)
    positives = max(int((y_train == 1).sum()), 1)
    negatives = int((y_train == 0).sum())
    imbalance_weight = negatives / positives

    if model_name == "Logistic Regression":
        estimator = LogisticRegression(
            class_weight="balanced", C=0.5, max_iter=2000,
            solver="liblinear", random_state=RANDOM_STATE
        )
    elif model_name == "Random Forest":
        estimator = RandomForestClassifier(
            n_estimators=350, max_depth=None, min_samples_leaf=2,
            class_weight="balanced_subsample", random_state=RANDOM_STATE,
            n_jobs=-1
        )
    else:
        estimator = XGBClassifier(
            objective="binary:logistic", eval_metric="logloss",
            n_estimators=350, learning_rate=0.03, max_depth=3,
            min_child_weight=3, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=2,
            scale_pos_weight=imbalance_weight,
            random_state=RANDOM_STATE, n_jobs=-1, tree_method="hist"
        )
    pipeline = Pipeline([("preprocessor", prep), ("model", estimator)])
    return pipeline, numeric_cols, categorical_cols

def churn_rate_by(frame, col):
    result = frame.groupby(col, observed=False)["Churn"].apply(
        lambda values: (values == "Yes").mean() * 100
    ).sort_values(ascending=False)
    return result

def make_recommendations(frame):
    recommendations = []
    for col, action in [
        ("Contract", "Test contract-specific renewal offers and explain the value of longer-term plans."),
        ("TenureGroup", "Strengthen onboarding, proactive check-ins, and early-life customer support."),
        ("PaymentMethod", "Review billing friction and make convenient payment options easy to adopt."),
        ("InternetService", "Review service quality, reliability, and plan fit for this customer segment."),
        ("TechSupport", "Test proactive technical support education and accessible troubleshooting."),
        ("OnlineSecurity", "Evaluate whether security education or relevant service bundles address customer needs.")
    ]:
        if col in frame.columns:
            rates = churn_rate_by(frame, col)
            if len(rates):
                segment = str(rates.index[0])
                rate = float(rates.iloc[0])
                recommendations.append({
                    "Insight": f"{col}: '{segment}' has the highest observed churn ({rate:.1f}%).",
                    "Recommendation": action
                })
    return recommendations

def get_customer_schema(frame, feature_columns):
    return frame[feature_columns].copy()

# Sidebar: model setup.
with st.sidebar:
    st.header("Dataset")
    st.caption("Using the Telco_Customer_Churn.csv dataset.")
    st.divider()
    st.header("Prediction model")
    selected_model_name = st.selectbox(
        "Choose algorithm",
        ["Logistic Regression", "Random Forest", "XGBoost"]
    )

try:
    raw_data = load_dataset()
    data = validate_data(raw_data)
except Exception as exc:
    st.error(str(exc))
    st.stop()

# Dataset insights / EDA
data["TenureGroup"] = pd.cut(
    data["tenure"],
    bins=[-1, 12, 24, 48, 60, 1000],
    labels=["0-12 months", "13-24 months", "25-48 months", "49-60 months", "60+ months"]
)

churn_numeric = data["Churn"].eq("Yes").astype(int)
overall_churn = float(churn_numeric.mean())
total_customers = len(data)
churned_customers = int(churn_numeric.sum())
avg_tenure = float(data["tenure"].mean())
avg_monthly = float(data["MonthlyCharges"].mean())

st.subheader("1. Dataset overview")
k1, k2, k3, k4 = st.columns(4)
k1.metric("Total customers", f"{total_customers:,}")
k2.metric("Churned customers", f"{churned_customers:,}")
k3.metric("Overall churn rate", f"{overall_churn:.1%}")
k4.metric("Average monthly charge", f"${avg_monthly:,.2f}")

with st.expander("Preview dataset and data quality", expanded=False):
    st.dataframe(data.head(20), use_container_width=True)
    quality = pd.DataFrame({
        "Missing values": data.isna().sum(),
        "Data type": data.dtypes.astype(str)
    })
    st.dataframe(quality, use_container_width=True)

st.subheader("2. Key insights from the existing dataset")
left, right = st.columns(2)

with left:
    contract_rates = churn_rate_by(data, "Contract")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(contract_rates.index.astype(str), contract_rates.values)
    ax.set_title("Churn rate by contract")
    ax.set_ylabel("Churn rate (%)")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=15)
    st.pyplot(fig)
    plt.close(fig)

with right:
    tenure_rates = churn_rate_by(data, "TenureGroup")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(tenure_rates.index.astype(str), tenure_rates.values)
    ax.set_title("Churn rate by customer tenure")
    ax.set_ylabel("Churn rate (%)")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=15)
    st.pyplot(fig)
    plt.close(fig)

left, right = st.columns(2)
with left:
    payment_rates = churn_rate_by(data, "PaymentMethod")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(payment_rates.index.astype(str), payment_rates.values)
    ax.set_title("Churn rate by payment method")
    ax.set_ylabel("Churn rate (%)")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=20)
    st.pyplot(fig)
    plt.close(fig)

with right:
    fig, ax = plt.subplots(figsize=(7, 4))
    no_churn = data.loc[data["Churn"] == "No", "MonthlyCharges"].dropna()
    churn = data.loc[data["Churn"] == "Yes", "MonthlyCharges"].dropna()
    ax.boxplot([no_churn, churn], tick_labels=["No churn", "Churn"])
    ax.set_title("Monthly charges by churn status")
    st.pyplot(fig)
    plt.close(fig)

st.markdown("#### Automatically identified segment patterns")
insights = []
for col in ["Contract", "TenureGroup", "PaymentMethod", "InternetService", "TechSupport", "OnlineSecurity"]:
    if col in data.columns:
        rates = churn_rate_by(data, col)
        if len(rates):
            insights.append({
                "Feature": col,
                "Highest-churn segment": str(rates.index[0]),
                "Observed churn rate": f"{rates.iloc[0]:.1f}%"
            })
st.dataframe(pd.DataFrame(insights), use_container_width=True, hide_index=True)

st.subheader("3. Recommendations based on the observed insights")
recommendations = make_recommendations(data)
for idx, item in enumerate(recommendations, start=1):
    with st.container(border=True):
        st.markdown(f"**{idx}. {item['Insight']}**")
        st.write(item["Recommendation"])
st.caption("These are associations in the historical dataset, not proof that a segment characteristic causes churn. Test retention actions before scaling them.")

# Model training / load
st.subheader("4. Train or load a churn prediction model")
artifact = None
if ARTIFACT_PATH.exists():
    try:
        artifact = joblib.load(ARTIFACT_PATH)
        st.info(f"Found saved model: {artifact.get('model_name', 'Saved model')}. You can use it below or train a new one.")
    except Exception as exc:
        st.warning(f"Saved model could not be loaded: {exc}")

if "trained_artifact" in st.session_state:
    artifact = st.session_state["trained_artifact"]

if st.button(f"Train {selected_model_name}", type="primary"):
    with st.spinner(f"Training {selected_model_name}..."):
        feature_columns = [c for c in data.columns if c not in ["Churn", "customerID", "TenureGroup"]]
        X_all = data[feature_columns].copy()
        y_all = data["Churn"].map({"No": 0, "Yes": 1}).astype(int)
        X_train, X_test, y_train, y_test = train_test_split(
            X_all, y_all, test_size=0.20, stratify=y_all, random_state=RANDOM_STATE
        )
        model, numeric_cols, categorical_cols = create_model(selected_model_name, X_train, y_train)
        model.fit(X_train, y_train)
        probabilities = model.predict_proba(X_test)[:, 1]
        threshold = 0.5
        predictions = (probabilities >= threshold).astype(int)

        new_artifact = {
            "model": model,
            "model_name": selected_model_name,
            "threshold": threshold,
            "feature_columns": feature_columns,
            "numeric_features": numeric_cols,
            "categorical_features": categorical_cols,
            "target": "Churn"
        }
        st.session_state["trained_artifact"] = new_artifact
        st.session_state["training_metrics"] = {
            "Accuracy": accuracy_score(y_test, predictions),
            "Precision": precision_score(y_test, predictions, zero_division=0),
            "Recall": recall_score(y_test, predictions, zero_division=0),
            "F1": f1_score(y_test, predictions, zero_division=0),
            "ROC-AUC": roc_auc_score(y_test, probabilities),
            "PR-AUC": average_precision_score(y_test, probabilities)
        }
        artifact = new_artifact

if artifact is not None:
    st.success(f"Ready to predict with {artifact.get('model_name', 'selected model')}.")
    metrics = st.session_state.get("training_metrics")
    if metrics:
        st.markdown("#### Hold-out evaluation for the model trained in this session")
        metric_cols = st.columns(3)
        for col, key in zip(metric_cols, ["ROC-AUC", "PR-AUC", "F1"]):
            col.metric(key, f"{metrics[key]:.3f}")
    st.download_button(
        "Download trained model artifact",
        data=joblib.dumps(artifact) if hasattr(joblib, "dumps") else __import__("pickle").dumps(artifact),
        file_name="best_churn_model.joblib",
        mime="application/octet-stream"
    )

    st.subheader("5. Predict churn for a new customer")
    st.write("Enter the customer's details below. The form uses the same feature columns as the training data.")
    feature_columns = artifact["feature_columns"]
    schema = data[feature_columns]

    with st.form("new_customer_form"):
        customer_values = {}
        widget_cols = st.columns(2)
        for index, col_name in enumerate(feature_columns):
            series = schema[col_name]
            with widget_cols[index % 2]:
                if pd.api.types.is_numeric_dtype(series):
                    default_num = float(series.median()) if series.notna().any() else 0.0
                    customer_values[col_name] = st.number_input(
                        col_name, value=default_num, step=1.0 if col_name in ["tenure", "SeniorCitizen"] else 0.5
                    )
                else:
                    choices = sorted(series.dropna().astype(str).unique().tolist())
                    if choices:
                        customer_values[col_name] = st.selectbox(col_name, choices)
                    else:
                        customer_values[col_name] = st.text_input(col_name)
        submitted = st.form_submit_button("Predict customer churn risk", type="primary")

    if submitted:
        customer_frame = pd.DataFrame([customer_values], columns=feature_columns)
        probability = float(artifact["model"].predict_proba(customer_frame)[:, 1][0])
        threshold = float(artifact.get("threshold", 0.5))
        prediction = int(probability >= threshold)

        a, b = st.columns(2)
        a.metric("Estimated churn probability", f"{probability:.1%}")
        b.metric("Decision threshold", f"{threshold:.2f}")

        if prediction == 1:
            st.error("Prediction: Higher churn risk")
            st.write("Suggested next step: review the customer's recent experience and consider a relevant retention conversation.")
        else:
            st.success("Prediction: Lower churn risk")
            st.write("Suggested next step: continue normal engagement and monitor changes in customer behaviour.")
        st.progress(min(max(probability, 0.0), 1.0))
        st.caption("This is a model estimate, not a certainty. Use it as decision support, not as the sole basis for customer treatment.")
else:
    st.info("Click **Train [model]** above to train a model before using the new-customer prediction form.")

st.divider()
st.caption("Project: Telco_Customer_Churn | Models: Logistic Regression, Random Forest, XGBoost")
