import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score
from mlxtend.plotting import plot_decision_regions
from sklearn.decomposition import PCA
import warnings

warnings.filterwarnings("ignore")

st.title("Spambase Dataset Analysis")

# Load dataset and prepare the data
pattern = re.compile(r'^(?!\|)(.+?):')
column_names = []

with open("spambase.names", "r") as file:
    for line in file:
        line = line.strip()
        match = pattern.match(line)
        if match:
            name = match.group(1).strip()
            if name.startswith("word_freq_") or name.startswith("char_freq_") or name.startswith("capital_run_length_"):
                column_names.append(name)

if "spam" not in column_names:
    column_names.append("spam")

data = pd.read_csv("spambase.data", delimiter=',', header=None, names=column_names)

st.subheader("Sample Data")
st.write(data.sample(10, random_state=42))

st.subheader("Unique Values and Shape")
st.write("Number of unique values per column:")
st.write(data.nunique())
st.write("Shape of the dataset:", data.shape)

st.subheader("Data Description")
st.write(data.describe())

data = data.dropna(subset=["spam"])

# Final data preparation
target = 'spam'
features = [i for i in data.columns if i != target]
X = data[features]
y = data[target]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scaling the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Logistic Regression Model
log_reg_2 = LogisticRegression(max_iter=1000)
log_reg_2.fit(X_train_scaled, y_train)

# Decision Tree Model
decision_tree_2 = DecisionTreeClassifier(max_depth=5)
decision_tree_2.fit(X_train_scaled, y_train)

# KNN Model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

# Predictions
y_pred_log_reg_2 = log_reg_2.predict(X_test_scaled)
y_pred_dt_2 = decision_tree_2.predict(X_test_scaled)
y_pred_knn = knn.predict(X_test_scaled)

# AUC Scores
st.subheader("Model Performance")
data = {
    'Classifier': ['Logistic Regression', 'Decision Tree', 'KNN'],
    'AUC (train)': [roc_auc_score(y_train, log_reg_2.predict(X_train_scaled)), 
                    roc_auc_score(y_train, decision_tree_2.predict(X_train_scaled)),
                    roc_auc_score(y_train, knn.predict(X_train_scaled))],
    'AUC (test)': [roc_auc_score(y_test, y_pred_log_reg_2), 
                   roc_auc_score(y_test, y_pred_dt_2), 
                   roc_auc_score(y_test, y_pred_knn)]
}
auc_df = pd.DataFrame(data)
st.write(auc_df)

# PCA for decision boundaries (reduce to 2D)
pca = PCA(n_components=2)
X_train_2d = pca.fit_transform(X_train_scaled)
X_test_2d = pca.transform(X_test_scaled)

# Model selection for decision boundary plot
model_choice = st.selectbox("Choose a model", ["Logistic Regression", "Decision Tree", "KNN"])

if model_choice == "Logistic Regression":
    fig, ax = plt.subplots(figsize=(8, 6))
    plot_decision_regions(X_test_2d, y_test.values, clf=log_reg_2)
    plt.title("Logistic Regression Decision Boundary (PCA)")
    st.pyplot(fig)
elif model_choice == "Decision Tree":
    fig, ax = plt.subplots(figsize=(8, 6))
    plot_decision_regions(X_test_2d, y_test.values, clf=decision_tree_2)
    plt.title("Decision Tree Decision Boundary (PCA)")
    st.pyplot(fig)
elif model_choice == "KNN":
    fig, ax = plt.subplots(figsize=(8, 6))
    plot_decision_regions(X_test_2d, y_test.values, clf=knn)
    plt.title("KNN Decision Boundary (PCA)")
    st.pyplot(fig)
