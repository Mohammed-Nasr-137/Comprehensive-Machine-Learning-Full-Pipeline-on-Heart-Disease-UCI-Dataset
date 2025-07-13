import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    roc_auc_score,
    precision_recall_curve
)
from mpl_toolkits.mplot3d import Axes3D
from io import StringIO
from dataset_handler import process_file
import sys
from pathlib import Path

# Add parent directory to path to import config
sys.path.append(str(Path(__file__).parent.parent))
from config import FINAL_VOTING_MODEL_PATH, EXPECTED_FEATURES_PATH, FINAL_VOTING_MODEL9_PATH

st.set_page_config(layout="wide")
st.title("💖 Heart Disease ML Project Dashboard")

# Load trained model
model = joblib.load(FINAL_VOTING_MODEL9_PATH)

# Sidebar
section = st.sidebar.selectbox("Select Section", ["Supervised Learning", "Unsupervised Clustering"])

uploaded_file = st.sidebar.file_uploader("Upload a UCI heart-disease .data file (e.g., processed.switzerland.data)", type=["data", "csv", "txt"])

if uploaded_file:
    df = process_file(uploaded_file)
    features = df.drop(columns=["target", "Prediction"])
    target = df["target"]
    predictions = df["Prediction"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    if section == "Supervised Learning":
        st.header("🔎 Supervised Classification")

        st.subheader("📋 Predictions Preview")
        st.dataframe(df[["target", "Prediction"]].head())

        st.download_button("Download Predictions CSV", df.to_csv(index=False), file_name="predictions.csv")

        st.subheader("📊 Classification Report")
        report = classification_report(target, predictions, output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose())

        st.subheader("Confusion Matrix")
        cm = confusion_matrix(target, predictions)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        st.pyplot(fig)

        st.subheader("ROC Curve")
        # Ensure correct feature alignment
        with open(EXPECTED_FEATURES_PATH) as f:
            expected_features = [line.strip() for line in f]

        for col in expected_features:
            if col not in features.columns:
                features[col] = 0.0

        features = features[expected_features]


        probas = model.predict_proba(features)[:, 1]
        fpr, tpr, _ = roc_curve(target, probas)
        auc = roc_auc_score(target, probas)
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
        ax.plot([0, 1], [0, 1], linestyle='--')
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve")
        ax.legend()
        st.pyplot(fig)

        st.subheader("Precision-Recall Curve")
        precision, recall, _ = precision_recall_curve(target, probas)
        fig, ax = plt.subplots()
        ax.plot(recall, precision, color='purple')
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall Curve")
        st.pyplot(fig)

    elif section == "Unsupervised Clustering":
        st.header("🔍 K-Means vs. Hierarchical Clustering (K=2)")

        # Clustering
        kmeans = KMeans(n_clusters=2, random_state=42)
        k_labels = kmeans.fit_predict(X_scaled)

        hc = AgglomerativeClustering(n_clusters=2, metric='euclidean', linkage='ward')
        hc_labels = hc.fit_predict(X_scaled)

        # PCA 3D
        pca = PCA(n_components=3)
        X_pca = pca.fit_transform(X_scaled)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("K-Means (3D PCA)")
            fig = plt.figure(figsize=(7, 5))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=k_labels, cmap='Set1', alpha=0.7)
            ax.set_xlabel("PCA 1")
            ax.set_ylabel("PCA 2")
            ax.set_zlabel("PCA 3")
            st.pyplot(fig)

        with col2:
            st.subheader("Hierarchical Clustering (3D PCA)")
            fig = plt.figure(figsize=(7, 5))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=hc_labels, cmap='Set2', alpha=0.7)
            ax.set_xlabel("PCA 1")
            ax.set_ylabel("PCA 2")
            ax.set_zlabel("PCA 3")
            st.pyplot(fig)

        st.subheader("Clustering vs. Actual Label")
        cm_k = confusion_matrix(target, k_labels)
        cm_h = confusion_matrix(target, hc_labels)

        st.write("**K-Means Confusion Matrix**")
        st.dataframe(pd.DataFrame(cm_k))

        st.write("**Hierarchical Confusion Matrix**")
        st.dataframe(pd.DataFrame(cm_h))
