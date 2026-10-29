import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix, classification_report

st.set_page_config(page_title="Mobile Price Classifier", layout="wide")
st.title("📱 Mobile Price Classification App")

model_mapping = {
    "Logistic Regression": "model/logistic_regression.pkl",
    "Decision Tree": "model/decision_tree.pkl",
    "KNN": "model/knn.pkl",
    "Naive Bayes": "model/naive_bayes.pkl",
    "Support Vector Machine": "model/support_vector_machine.pkl"
}

selected_model = st.sidebar.selectbox("Select ML Model", list(model_mapping.keys()))
model = joblib.load(model_mapping[selected_model])
scaler = joblib.load("model/scaler.pkl")

uploaded_file = st.sidebar.file_uploader("Upload test_data.csv", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    X = df.drop(columns=["price_range"])
    y = df["price_range"]
    
    if selected_model in ["Logistic Regression", "KNN", "Support Vector Machine"]:
        X = scaler.transform(X)
        
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X) if hasattr(model, "predict_proba") else None
    
    st.subheader(f"Metrics for {selected_model}")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Accuracy", f"{accuracy_score(y, y_pred):.4f}")
    if y_prob is not None:
        c2.metric("AUC", f"{roc_auc_score(y, y_prob, multi_class='ovr', average='weighted'):.4f}")
    c3.metric("Precision", f"{precision_score(y, y_pred, average='weighted', zero_division=0):.4f}")
    c4.metric("Recall", f"{recall_score(y, y_pred, average='weighted', zero_division=0):.4f}")
    c5.metric("F1 Score", f"{f1_score(y, y_pred, average='weighted', zero_division=0):.4f}")
    c6.metric("MCC", f"{matthews_corrcoef(y, y_pred):.4f}")

    col1, col2 = st.columns(2)
    with col1:
        st.write("### Confusion Matrix")
        fig, ax = plt.subplots(figsize=(4, 3))
        sns.heatmap(confusion_matrix(y, y_pred), annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax)
        st.pyplot(fig)
    with col2:
        st.write("### Classification Report")
        report = pd.DataFrame(classification_report(y, y_pred, output_dict=True)).transpose()
        st.dataframe(report.style.format("{:.3f}"))
else:
    st.info("Upload test_data.csv in the sidebar to view results.")
