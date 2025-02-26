import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from mlxtend.plotting import plot_decision_regions
from sklearn.metrics import roc_curve, roc_auc_score
import warnings

warnings.filterwarnings("ignore")

# Streamlit UI
st.title('Spam Classification with Machine Learning')
st.write("This app uses a classification model to predict spam messages from the Spambase dataset.")

# Read the dataset directly from files
data = pd.read_csv("spambase.data", delimiter=',', header=None)

# Extract column names from 'spambase.names' file
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

data.columns = column_names

# Preprocessing
data = data.dropna(subset=["spam"])
unique_classes = data['spam'].unique()
if len(unique_classes) > 2:
    threshold = np.median(data['spam'])
    data['binary_label'] = (data['spam'] > threshold).astype(int)
else:
    st.write("Binary classification already present.")

major_class = data['spam'].value_counts().idxmax()
data['spam'] = (data['spam'] == major_class).astype(int)

# Fill missing values with column mean
data_pos = data[data["spam"] == 1]
data_neg = data[data["spam"] == 0]
data_pos = data_pos.fillna(data_pos.mean())
data_neg = data_neg.fillna(data_neg.mean())
data = pd.concat([data_pos, data_neg]).sort_index()

# Prepare features and target
target = 'spam'
features = [i for i in data.columns if i != target]
X = data[features]
y = data[target]

# Feature correlation and selection
features_up_10_unique = X.loc[:, X.nunique() > 10]
correlations = features_up_10_unique.corrwith(y).abs()
top_features = correlations.nlargest(3)
features_to_plot = top_features.index

# 3D Plot
fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[y == 1][features_to_plot[0]], X[y == 1][features_to_plot[1]], X[y == 1][features_to_plot[2]], c='r', marker='o', label='Spam')
ax.scatter(X[y == 0][features_to_plot[0]], X[y == 0][features_to_plot[1]], X[y == 0][features_to_plot[2]], c='g', marker='^', label='Not Spam')
ax.set_xlabel(features_to_plot[0])
ax.set_ylabel(features_to_plot[1])
ax.set_zlabel(features_to_plot[2])
ax.set_title('Spambase Dataset - 3D Visualization')
ax.legend()
st.pyplot(fig)

# Train-Test Split and Scaling
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train models
knn_2 = KNeighborsClassifier(n_neighbors=3)
log_reg_2 = LogisticRegression(max_iter=565)
decision_tree_2 = DecisionTreeClassifier(max_depth=5)

knn_2.fit(X_train_scaled, y_train)
log_reg_2.fit(X_train_scaled, y_train)
decision_tree_2.fit(X_train_scaled, y_train)

# Predictions
y_prob_knn_2 = knn_2.predict_proba(X_test_scaled)[:, 1]
y_prob_log_reg_2 = log_reg_2.predict_proba(X_test_scaled)[:, 1]
y_prob_dt_2 = decision_tree_2.predict_proba(X_test_scaled)[:, 1]

# ROC AUC Scores
roc_auc_knn_2 = roc_auc_score(y_test, y_prob_knn_2)
roc_auc_log_reg_2 = roc_auc_score(y_test, y_prob_log_reg_2)
roc_auc_dt_2 = roc_auc_score(y_test, y_prob_dt_2)

# Display AUC Scores
st.write(f'KNN AUC: {roc_auc_knn_2:.2f}')
st.write(f'Logistic Regression AUC: {roc_auc_log_reg_2:.2f}')
st.write(f'Decision Tree AUC: {roc_auc_dt_2:.2f}')

# ROC Curves
fpr_knn_2, tpr_knn_2, _ = roc_curve(y_test, y_prob_knn_2)
fpr_log_reg_2, tpr_log_reg_2, _ = roc_curve(y_test, y_prob_log_reg_2)
fpr_dt_2, tpr_dt_2, _ = roc_curve(y_test, y_prob_dt_2)

fig_roc = plt.figure(figsize=(10, 8))
plt.plot(fpr_knn_2, tpr_knn_2, color='blue', lw=2, label=f'KNN (AUC = {roc_auc_knn_2:.2f})')
plt.plot(fpr_log_reg_2, tpr_log_reg_2, color='green', lw=2, label=f'Logistic Regression (AUC = {roc_auc_log_reg_2:.2f})')
plt.plot(fpr_dt_2, tpr_dt_2, color='red', lw=2, label=f'Decision Tree (AUC = {roc_auc_dt_2:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=2)
plt.title('ROC Curves for Classifiers', fontsize=16)
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.legend(loc='lower right', fontsize=12)
plt.grid(True)
st.pyplot(fig_roc)

# Display AUC Table
auc_data = {
    'Classifier': ['KNN', 'Logistic Regression', 'Decision Tree'],
    'AUC (test)': [roc_auc_knn_2, roc_auc_log_reg_2, roc_auc_dt_2]
}
auc_df = pd.DataFrame(auc_data)
st.write(auc_df)
