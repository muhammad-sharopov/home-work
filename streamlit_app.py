import streamlit as st
import re
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

st.title('📊 Spambase Analysis & Classification')

# Загрузка данных
data = pd.read_csv("spambase.data", delimiter=',', header=None)

# Названия колонок
def get_column_names():
    pattern = re.compile(r'^(?!\|)(.+?):')
    column_names = []
    with open("spambase.names", "r") as file:
        for line in file:
            match = pattern.match(line.strip())
            if match:
                name = match.group(1).strip()
                if name.startswith("word_freq_") or name.startswith("char_freq_") or name.startswith("capital_run_length_"):
                    column_names.append(name)
    column_names.append("spam")
    return column_names

data.columns = get_column_names()
st.write("### Sample Data")
st.dataframe(data.sample(10))

# Разделение данных
X = data.drop(columns=['spam'])
y = data['spam']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Стандартизация
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Обучение моделей
models = {
    'Logistic Regression': LogisticRegression(max_iter=565),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'Decision Tree': DecisionTreeClassifier(max_depth=5)
}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)

# Оценка моделей
st.write("### Model Performance")
auc_scores = {}
for name, model in models.items():
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    auc_score = roc_auc_score(y_test, y_prob)
    auc_scores[name] = auc_score

df_auc = pd.DataFrame(list(auc_scores.items()), columns=['Model', 'AUC Score'])
st.dataframe(df_auc)

# ROC-кривые
st.write("### ROC Curves")
plt.figure(figsize=(10, 6))
for name, model in models.items():
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {auc_scores[name]:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.legend()
st.pyplot(plt)
