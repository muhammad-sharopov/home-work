import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
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


st.subheader("Shape")

selected_view = st.radio("Select data view:", ("Rows", "Columns"))

if selected_view == "Rows":
    if "show_rows" not in st.session_state:
        st.session_state.show_rows = False

    st.write("Number of rows:", data.shape[0])

    if st.button("Show Rows"):
        st.session_state.show_rows = not st.session_state.show_rows 

    if st.session_state.show_rows:
        st.dataframe(data) 

elif selected_view == "Columns":
    if "show_columns" not in st.session_state:
        st.session_state.show_columns = False

    st.write("Number of columns:", data.shape[1])

    if st.button("Show Columns"):
        st.session_state.show_columns = not st.session_state.show_columns 

    if st.session_state.show_columns:
        st.write("Column names:")
        st.write(data.columns) 

st.subheader("Unique Values")

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
    st.write("Binarization for target not needed.")

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

st.subheader('Spambase Dataset - 3D Visualization')

selected_class = st.radio("Select the class to display:", ("Both", "Spam", "Not Spam"))

features_to_plot = top_features.index
X_top = data[features_to_plot]

fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(111, projection='3d')

if selected_class == "Spam":
    ax.scatter(X[y == 1][features_to_plot[0]], X[y == 1][features_to_plot[1]], X[y == 1][features_to_plot[2]],
               c='r', marker='o', label='Spam')
elif selected_class == "Not Spam":
    ax.scatter(X[y == 0][features_to_plot[0]], X[y == 0][features_to_plot[1]], X[y == 0][features_to_plot[2]],
               c='g', marker='^', label='Not Spam')
else:
    ax.scatter(X[y == 1][features_to_plot[0]], X[y == 1][features_to_plot[1]], X[y == 1][features_to_plot[2]],
               c='r', marker='o', label='Spam')
    ax.scatter(X[y == 0][features_to_plot[0]], X[y == 0][features_to_plot[1]], X[y == 0][features_to_plot[2]],
               c='g', marker='^', label='Not Spam')

ax.set_xlabel(features_to_plot[0])
ax.set_ylabel(features_to_plot[1])
ax.set_zlabel(features_to_plot[2])
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

selected_model = st.selectbox("Select a model for decision boundary:", ["Logistic Regression", "Decision Tree", "KNN"])

# Словарь для соответствия названий моделей объектам
model_dict = {
    "Logistic Regression": log_reg_2,
    "Decision Tree": decision_tree_2,
    "KNN": knn
}

if selected_model:
    model = model_dict[selected_model]
    
    # Генерируем границы решений
    x_min, x_max = X_test_2_scaled[:, 0].min() - 1, X_test_2_scaled[:, 0].max() + 1
    y_min, y_max = X_test_2_scaled[:, 1].min() - 1, X_test_2_scaled[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Создаем интерактивный график с Plotly
    fig = go.Figure(data=go.Heatmap(
        z=Z,
        x=np.linspace(x_min, x_max, Z.shape[1]),
        y=np.linspace(y_min, y_max, Z.shape[0]),
        colorscale="Viridis",
        opacity=0.5
    ))
    
    # Добавляем точки тестовой выборки
    fig.add_trace(go.Scatter(
        x=X_test_2_scaled[:, 0], 
        y=X_test_2_scaled[:, 1], 
        mode='markers', 
        marker=dict(color=y_test.values, colorscale='Jet', size=8),
        name="Test Data"
    ))

    # Настройка осей
    fig.update_layout(
        title=f"{selected_model} Decision Boundary",
        xaxis_title=top_2_features[0],
        yaxis_title=top_2_features[1],
        coloraxis_showscale=False
    )

    st.plotly_chart(fig, use_container_width=True)
