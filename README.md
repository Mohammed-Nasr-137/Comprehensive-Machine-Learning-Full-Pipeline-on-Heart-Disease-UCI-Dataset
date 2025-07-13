# Comprehensive-Machine-Learning-Full-Pipeline-on-Heart-Disease-UCI-Dataset
# 🩸 Heart Disease Prediction Web App

A full-stack machine learning project for detecting heart disease based on UCI datasets. Built with **Streamlit** and deployed via **Ngrok**, this app provides both an interactive frontend and a robust backend pipeline for medical data processing, model inference, and visualization.

---

## 🚀 Features

* ✅ Upload support for multiple `.data` files from UCI repository (Cleveland, Hungary, Switzerland)
* ✅ Automatic preprocessing (NaN handling, encoding, scaling, binarization)
* ✅ Feature selection (Random Forest, RFE, Union strategy)
* ✅ Voting Classifier (Logistic Regression + SVM + Random Forest)
* ✅ Performance visualizations: ROC Curve, Confusion Matrix
* ✅ Export predictions to CSV
* ✅ 3D Clustering visualization (KMeans, Hierarchical)
* ✅ Lightweight UI using Streamlit

---

## 🔧 Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/heart-disease-app.git
cd heart-disease-app
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🌐 Running the App

### Local launch:

```bash
streamlit run app.py
```

### Optional: Expose via Ngrok

```bash
ngrok http 8501
```

You will get a public link like: `https://abc123.ngrok-free.app`

---

## 🔍 Interacting with the App

### 1. Upload Dataset

* Click “Browse files”
* Upload any of:

  * `processed.cleveland.data`
  * `processed.hungarian.data`
  * `processed.switzerland.data`
  * Or your custom `.data` file (same format)

### 2. View Predictions

* Model automatically preprocesses the file
* Displays prediction statistics
* Confusion matrix and ROC curve appear

### 3. Export

* Click the “Download Predictions” button
* CSV will include original data + `Prediction` column

---

## 📊 Models and Pipeline

### Preprocessing

* Imputation: mean filling for missing values
* Encoding: one-hot for `cp`, `thal`, `slope`, `ca`, etc.
* Feature alignment: enforced via `expected_features.txt`
* Scaling: standardization on numerical features

### Feature Selection

* Random Forest top N (importance-based)
* RFE (Recursive Feature Elimination)
* Final union of both feature sets is used for training/inference

### Trained Models

* Logistic Regression (with tuning)
* Support Vector Machine
* Random Forest
* Combined using a Voting Classifier

### Metrics (on combined test set)

* Accuracy: \~85%
* ROC AUC: \~0.93
* Precision, Recall, F1 reported per class

---

## 🌎 Unsupervised Learning

Included in backend notebooks:

* K-Means (with elbow method, visualized in 3D)
* Agglomerative Clustering (Ward linkage)
* Evaluation: Adjusted Rand Index, Silhouette Score

---

## 🔍 Project Structure

```
project/
├── app.py                    # Streamlit frontend
├── dataset_handler.py       # Preprocessing & feature alignment
├── predict_new_dataset.py   # Model inference logic
├── final_voting_model3.pkl  # Pretrained ensemble model
├── expected_features.txt    # Feature schema during training
├── requirements.txt         # Python dependencies
├── data/                    # .data input files
│   ├── processed.cleveland.data
│   ├── processed.hungarian.data
│   └── processed.switzerland.data
```

---

## 🙌 Acknowledgments

* UCI Heart Disease Dataset: [https://archive.ics.uci.edu/ml/datasets/Heart+Disease](https://archive.ics.uci.edu/ml/datasets/Heart+Disease)

