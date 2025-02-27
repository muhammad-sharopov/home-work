import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve, roc_auc_score
from mlxtend.plotting import plot_decision_regions
import warnings
warnings.filterwarnings("ignore")

st.title("Spambase Dataset Analysis")

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

with st.expander('Sample Data'):
    st.write("Features (X)")
    X_raw = data.drop('spam', axis=1)
    st.dataframe(X_raw)

    st.write("Target (y)")
    y_raw = data['spam']
    st.dataframe(y_raw)


st.subheader("Unique Values and Shape")

st.write("Shape of the dataset:", data.shape)

st.subheader("Unique Values and Shape")

selected_column = st.selectbox("Select a column:", data.columns)

if selected_column:
    st.write(f"Number of unique values in '{selected_column}':", data[selected_column].nunique())
    
    with st.expander("Show Unique Values"):
        st.dataframe(sorted(data[selected_column].unique()))



st.subheader("Data Description")

selected_column = st.selectbox("Select a column to describe:", data.columns)

st.write(data[selected_column].describe())




data = data.dropna(subset=["spam"])

unique_classes = data['spam'].unique()
if len(unique_classes) > 2:
    threshold = np.median(data['spam'])
    data['binary_label'] = (data['spam'] > threshold).astype(int)
    st.subheader("Class Distribution After Binarization")
    st.write(data['binary_label'].value_counts())
else:
    st.write("Binarization not needed.")

major_class = data['spam'].value_counts().idxmax()
data['spam'] = (data['spam'] == major_class).astype(int)
st.subheader("Final Spam Distribution")
st.write(data['spam'].value_counts())

numeric_columns = data.columns[data.dtypes == 'object']
for col in numeric_columns:
    data[col] = pd.to_numeric(data[col], errors='coerce')

data = data.select_dtypes(include=['number'])

data_pos = data[data["spam"] == 1]
data_neg = data[data["spam"] == 0]
data_pos = data_pos.fillna(data_pos.mean())
data_neg = data_neg.fillna(data_neg.mean())
data = pd.concat([data_pos, data_neg]).sort_index()

st.subheader("Missing Values Count")
st.write(data.isnull().sum().sum())

target = 'spam'
features = [i for i in data.columns if i != target]
X = data[features]
y = data[target]

features_up_10_unique = X.loc[:, X.nunique() > 10]
correlations = features_up_10_unique.corrwith(y).abs()
top_features = correlations.nlargest(3)
st.subheader("Top Correlated Features")
st.write(top_features)

features_to_plot = top_features.index
X_top = data[features_to_plot]

fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[y == 1][features_to_plot[0]], X[y == 1][features_to_plot[1]], X[y == 1][features_to_plot[2]],
           c='r', marker='o', label='Spam')
ax.scatter(X[y == 0][features_to_plot[0]], X[y == 0][features_to_plot[1]], X[y == 0][features_to_plot[2]],
           c='g', marker='^', label='Not Spam')
ax.set_xlabel(features_to_plot[0])
ax.set_ylabel(features_to_plot[1])
ax.set_zlabel(features_to_plot[2])
ax.set_title('Spambase Dataset - 3D Visualization')
ax.legend()
st.pyplot(fig)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

top_2_features = top_features.head(2).index
X_2 = data[top_2_features]
X_train_2, X_test_2, y_train, y_test = train_test_split(X_2, y, test_size=0.3, random_state=42)
X_train_2_scaled = scaler.fit_transform(X_train_2)
X_test_2_scaled = scaler.transform(X_test_2)

log_reg_2 = LogisticRegression(max_iter=565)
log_reg_2.fit(X_train_2_scaled, y_train)
y_pred_log_reg_2 = log_reg_2.predict(X_test_2_scaled)

decision_tree_2 = DecisionTreeClassifier(max_depth=5)
decision_tree_2.fit(X_train_2_scaled, y_train)
y_pred_dt_2 = decision_tree_2.predict(X_test_2_scaled)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_2_scaled, y_train)
y_pred_knn = knn.predict(X_test_2_scaled)

y_prob_log_reg_train = log_reg_2.predict_proba(X_train_2_scaled)[:, 1]
y_prob_dt_train = decision_tree_2.predict_proba(X_train_2_scaled)[:, 1]
y_prob_knn_train = knn.predict_proba(X_train_2_scaled)[:, 1]

y_prob_log_reg_test = log_reg_2.predict_proba(X_test_2_scaled)[:, 1]
y_prob_dt_test = decision_tree_2.predict_proba(X_test_2_scaled)[:, 1]
y_prob_knn_test = knn.predict_proba(X_test_2_scaled)[:, 1]

roc_auc_log_reg_train = roc_auc_score(y_train, y_prob_log_reg_train)
roc_auc_dt_train = roc_auc_score(y_train, y_prob_dt_train)
roc_auc_knn_train = roc_auc_score(y_train, y_prob_knn_train)

roc_auc_log_reg_test = roc_auc_score(y_test, y_prob_log_reg_test)
roc_auc_dt_test = roc_auc_score(y_test, y_prob_dt_test)
roc_auc_knn_test = roc_auc_score(y_test, y_prob_knn_test)

data = {
    'Classifier': ['Logistic Regression', 'Decision Tree', 'KNN'],
    'AUC (train)': [roc_auc_log_reg_train, roc_auc_dt_train, roc_auc_knn_train],
    'AUC (test)': [roc_auc_log_reg_test, roc_auc_dt_test, roc_auc_knn_test]
}
auc_df = pd.DataFrame(data)
st.subheader("Model Performance")
st.write(auc_df)

st.subheader("Decision Boundaries")
fig, ax = plt.subplots(figsize=(8, 6))
plot_decision_regions(X_test_2_scaled, y_test.values, clf=log_reg_2)
plt.title("Logistic Regression Decision Boundary")
st.pyplot(fig)

fig, ax = plt.subplots(figsize=(8, 6))
plot_decision_regions(X_test_2_scaled, y_test.values, clf=decision_tree_2)
plt.title("Decision Tree Decision Boundary")
st.pyplot(fig)

fig, ax = plt.subplots(figsize=(8, 6))
plot_decision_regions(X_test_2_scaled, y_test.values, clf=knn)
plt.title("KNN Decision Boundary")
st.pyplot(fig)
