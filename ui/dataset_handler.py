import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import joblib
import sys
from pathlib import Path

# Add parent directory to path to import config
sys.path.append(str(Path(__file__).parent.parent))
from config import FINAL_VOTING_MODEL_PATH, EXPECTED_FEATURES_PATH

def load_uci_data(file_path):
    column_names = [
        "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
        "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"
    ]
    df = pd.read_csv(file_path, header=None, names=column_names)

    # Convert "?" to NaN
    df.replace("?", np.nan, inplace=True)
    df = df.apply(pd.to_numeric, errors='coerce')

    return df


def preprocess_data(data_frame):
    data_frame = data_frame.dropna(subset=["target"])

    imputer = SimpleImputer(strategy="mean")
    data_frame[:] = imputer.fit_transform(data_frame)

    categorical_cols = ['cp', 'restecg', 'slope', 'thal', 'ca']
    df_encoded = pd.get_dummies(data_frame, columns=categorical_cols, drop_first=True)
    df_encoded['target'] = df_encoded['target'].apply(lambda x: 0 if x == 0 else 1)

    numeric_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    scaler = StandardScaler()
    df_encoded[numeric_cols] = scaler.fit_transform(df_encoded[numeric_cols])

    return df_encoded

"""
def feature_selector(data_frame):
    X = data_frame.drop(columns=['target'])
    y = data_frame['target']

    # RF
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    importances = model.feature_importances_
    feature_names = X.columns
    feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)

    # RFE
    estimator = LogisticRegression(max_iter=1000)
    selector = RFE(estimator, n_features_to_select=10)
    selector = selector.fit(X, y)
    selected_features = X.columns[selector.support_]

    rf_top_n = feat_imp.head(15).index.tolist()
    rfe_selected = selected_features.tolist()
    union_features = list(set(rf_top_n).union(set(rfe_selected)))
    X_union = X[union_features]
    X_union['target'] = y

    return X_union
"""


def align_features_for_prediction(df):
    with open(EXPECTED_FEATURES_PATH) as f:
        expected_features = [line.strip() for line in f]

    for col in expected_features:
        if col not in df.columns:
            df[col] = 0.0

    return df[expected_features]



def process_file(filepath):
    data = load_uci_data(filepath)
    encoded_data = preprocess_data(data)
    X = encoded_data.drop(columns=["target"])
    y = encoded_data["target"]

    X_aligned = align_features_for_prediction(X)

    model = joblib.load(FINAL_VOTING_MODEL_PATH)
    encoded_data["Prediction"] = model.predict(X_aligned)

    return encoded_data
