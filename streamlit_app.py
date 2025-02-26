import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D as ax
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from mlxtend.plotting import plot_decision_regions
from sklearn.metrics import roc_curve, roc_auc_score
import streamlit as st
import warnings

warnings.filterwarnings("ignore")

# Чтение данных и установка имен колонок
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

# Streamlit: Отображаем уникальные значения
st.subheader("Уникальные значения")
st.write(data.nunique())

# Описание данных
st.subheader("Описание данных")
st.write(data.describe())

# Убираем строки с пропусками
data = data.dropna(subset=["spam"])

# Преобразуем бинарные классы
unique_classes = data['spam'].unique()
if len(unique_classes) > 2:
    threshold = np.median(data['spam'])
    data['binary_label'] = (data['spam'] > threshold).astype(int)

# Обрабатываем числовые данные
numeric_columns = data.columns[data.dtypes == 'object']
for col in numeric_columns:
    data[col] = pd.to_numeric(data[col], errors='coerce')

data = data.select_dtypes(include=['number'])

# Разделяем данные по классам
data_pos = data[data["spam"] == 1]
data_neg = data[data["spam"] == 0]

# Заполняем пропуски средними значениями
data_pos = data_pos.fillna(data_pos.mean())
data_neg = data_neg.fillna(data_neg.mean())

data = pd.concat([data_pos, data_neg]).sort_index()

# Отбираем нужные признаки
target = 'spam'
features = [i for i in data.columns if i != target]
X = data[features]
y = data[target]

# Определяем топ 3 признаков
correlations = X.corrwith(y).abs()
top_features = correlations.nlargest(3)
features_to_plot = top_features.index

X_top = data[features_to_plot]

# 3D график
st.subheader('3D График данных')
fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(X_top[y == 1][features_to_plot[0]], X_top[y == 1][features_to_plot[1]], X_top[y == 1][features_to_plot[2]],
           c='r', marker='o', label='Spam')
ax.scatter(X_top[y == 0][features_to_plot[0]], X_top[y == 0][features_to_plot[1]], X_top[y == 0][features_to_plot[2]],
           c='g', marker='^', label='Not Spam')

ax.set_xlabel(features_to_plot[0])
ax.set_ylabel(features_to_plot[1])
ax.set_zlabel(features_to_plot[2])

ax.set_title('Spambase Dataset - 3D Visualization')

ax.legend()
st.pyplot(fig)

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Модели для обучения
models = {
    "Logistic Regression": LogisticRegression(max_iter=565),
    "KNN": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(max_depth=5)
}

# Обучаем модели и отображаем графики решений
st.subheader('Графики решений для моделей')

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    fig2 = plt.figure(figsize=(15, 5))
    plot_decision_regions(X_train_scaled, y_train.values, clf=model, legend=2)
    plt.title(f'{name} Decision Region')
    plt.xlabel(features_to_plot[0])
    plt.ylabel(features_to_plot[1])
    plt.tight_layout()
    st.pyplot(fig2)

    # Предсказания и оценка AUC
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = roc_auc_score(y_test, y_prob)

    st.subheader(f'ROC Кривая для {name}')
    fig3 = plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=2)
    plt.title(f'ROC Curve for {name}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.grid(True)
    st.pyplot(fig3)

    st.write(f'{name} AUC: {roc_auc:.2f}')
