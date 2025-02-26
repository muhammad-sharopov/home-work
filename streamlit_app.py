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
from sklearn.metrics import roc_curve, roc_auc_score, classification_report, confusion_matrix
from mlxtend.plotting import plot_decision_regions
import plotly.graph_objects as go
import joblib

# Заголовок приложения Streamlit
st.title("Spambase Dataset Analysis")

# Загрузка данных и подготовка колонок
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

# Отображение данных
st.subheader("Sample Data")
st.write(data.sample(10, random_state=42))

# Описание данных
st.subheader("Unique Values and Shape")
st.write("Number of unique values per column:")
st.write(data.nunique())
st.write("Shape of the dataset:", data.shape)

# Описание статистики
st.subheader("Data Description")
st.write(data.describe())

# Удаление пустых значений
data = data.dropna(subset=["spam"])

# Бинаризация классов
unique_classes = data['spam'].unique()
if len(unique_classes) > 2:
    threshold = np.median(data['spam'])
    data['binary_label'] = (data['spam'] > threshold).astype(int)
    st.subheader("Class Distribution After Binarization")
    st.write(data['binary_label'].value_counts())
else:
    st.write("Binarization not needed.")

# Финальная переработка классов
major_class = data['spam'].value_counts().idxmax()
data['spam'] = (data['spam'] == major_class).astype(int)
st.subheader("Final Spam Distribution")
st.write(data['spam'].value_counts())

# Преобразование категориальных признаков в числовые
numeric_columns = data.columns[data.dtypes == 'object']
for col in numeric_columns:
    data[col] = pd.to_numeric(data[col], errors='coerce')

data = data.select_dtypes(include=['number'])

# Обработка пропусков в данных
data_pos = data[data["spam"] == 1]
data_neg = data[data["spam"] == 0]
data_pos = data_pos.fillna(data_pos.mean())
data_neg = data_neg.fillna(data_neg.mean())
data = pd.concat([data_pos, data_neg]).sort_index()

st.subheader("Missing Values Count")
st.write(data.isnull().sum().sum())

# Разделение на целевую переменную и признаки
target = 'spam'
features = [i for i in data.columns if i != target]
X = data[features]
y = data[target]

# Корреляции признаков с целевой переменной
features_up_10_unique = X.loc[:, X.nunique() > 10]
correlations = features_up_10_unique.corrwith(y).abs()
top_features = correlations.nlargest(3)
st.subheader("Top Correlated Features")
st.write(top_features)

# Визуализация признаков
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

# Разделение на обучающие и тестовые данные
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Модели машинного обучения
log_reg_2 = LogisticRegression(max_iter=565)
log_reg_2.fit(X_train_scaled, y_train)

decision_tree_2 = DecisionTreeClassifier(max_depth=5)
decision_tree_2.fit(X_train_scaled, y_train)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

# Оценка качества моделей
st.subheader("Model Performance")
data = {
    'Classifier': ['Logistic Regression', 'Decision Tree', 'KNN'],
    'AUC (train)': [roc_auc_score(y_train, log_reg_2.predict(X_train_scaled)),
                     roc_auc_score(y_train, decision_tree_2.predict(X_train_scaled)),
                     roc_auc_score(y_train, knn.predict(X_train_scaled))],
    'AUC (test)': [roc_auc_score(y_test, log_reg_2.predict(X_test_scaled)),
                    roc_auc_score(y_test, decision_tree_2.predict(X_test_scaled)),
                    roc_auc_score(y_test, knn.predict(X_test_scaled))]
}
auc_df = pd.DataFrame(data)
st.write(auc_df)

# Визуализация границ решений
model_choice = st.selectbox("Choose a model", ["Logistic Regression", "Decision Tree", "KNN"])
if model_choice == "Logistic Regression":
    fig, ax = plt.subplots(figsize=(8, 6))
    plot_decision_regions(X_test_scaled, y_test.values, clf=log_reg_2)
    plt.title("Logistic Regression Decision Boundary")
    st.pyplot(fig)
elif model_choice == "Decision Tree":
    fig, ax = plt.subplots(figsize=(8, 6))
    plot_decision_regions(X_test_scaled, y_test.values, clf=decision_tree_2)
    plt.title("Decision Tree Decision Boundary")
    st.pyplot(fig)
elif model_choice == "KNN":
    fig, ax = plt.subplots(figsize=(8, 6))
    plot_decision_regions(X_test_scaled, y_test.values, clf=knn)
    plt.title("KNN Decision Boundary")
    st.pyplot(fig)

# Сохранение модели
joblib.dump(log_reg_2, 'logistic_regression_model.pkl')

# Загрузка модели
model = joblib.load('logistic_regression_model.pkl')

# Функция для отображения метрик модели
def display_metrics(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    st.subheader("Evaluation Metrics")
    st.metric("Accuracy", accuracy)
    st.metric("Precision", precision)
    st.metric("Recall", recall)
    st.metric("F1 Score", f1)

# Оценка метрик для модели логистической регрессии
display_metrics(log_reg_2, X_test_scaled, y_test)

# Визуализация матрицы ошибок
def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Not Spam", "Spam"], yticklabels=["Not Spam", "Spam"])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    st.pyplot(plt)

# Матрица ошибок для логистической регрессии
plot_confusion_matrix(y_test, log_reg_2.predict(X_test_scaled))

# Визуализация ROC-кривой
def plot_roc_curve(fpr, tpr, auc, model_name):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'{model_name} (AUC = {auc:.2f})'))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash', color='gray')))
    fig.update_layout(title='Receiver Operating Characteristic (ROC) Curve',
                      xaxis_title='False Positive Rate',
                      yaxis_title='True Positive Rate')
    st.plotly_chart(fig)

# ROC для логистической регрессии
fpr, tpr, thresholds = roc_curve(y_test, log_reg_2.predict(X_test_scaled))
auc = roc_auc_score(y_test, log_reg_2.predict(X_test_scaled))
plot_roc_curve(fpr, tpr, auc, 'Logistic Regression')
