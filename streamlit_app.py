import streamlit as st
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
import warnings
warnings.filterwarnings("ignore")

all_features = []
# Загрузка и предобработка данных (с кэшированием)
@st.cache_data
def load_and_preprocess_data():
    global all_features 
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

    # Предобработка данных (как в Colab)
    data = data.dropna(subset=["spam"])
    unique_classes = data['spam'].unique()
    if len(unique_classes) > 2:
        threshold = np.median(data['spam'])
        data['binary_label'] = (data['spam'] > threshold).astype(int)
    else:
        print("Бинарная классификация уже присутствует.")

    major_class = data['spam'].value_counts().idxmax()
    data['spam'] = (data['spam'] == major_class).astype(int)

    numeric_columns = data.columns[data.dtypes == 'object']
    for col in numeric_columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')

    data = data.select_dtypes(include=['number'])
    data_pos = data[data["spam"] == 1]
    data_neg = data[data["spam"] == 0]
    data_pos = data_pos.fillna(data_pos.mean())
    data_neg = data_neg.fillna(data_neg.mean())
    data = pd.concat([data_pos, data_neg]).sort_index()

    all_features = column_names
    return data

data = load_and_preprocess_data()

# Разделение данных и масштабирование
target = 'spam'
features = [i for i in data.columns if i != target]
X = data[features]
y = data[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Выбор признаков для 2D визуализации (top 2)
correlations = X.corrwith(y).abs()
top_features = correlations.nlargest(3)
features_to_plot = top_features.index

top_2_features = top_features.head(2).index
X_2 = data[top_2_features]
X_train_2, X_test_2, y_train, y_test = train_test_split(X_2, y, test_size=0.3, random_state=42)
X_train_2_scaled = scaler.fit_transform(X_train_2)
X_test_2_scaled = scaler.transform(X_test_2)

# Обучение моделей (вне Streamlit, с кэшированием)
@st.cache_resource
def train_models(X_train, y_train):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=565),
        "KNN": KNeighborsClassifier(),
        "Decision Tree": DecisionTreeClassifier(max_depth=5)
    }
    for name, model in models.items():
        model.fit(X_train, y_train)
    return models

models = train_models(X_train_2_scaled, y_train)

# Streamlit interface
st.title('Spambase Classification')

# 3D Visualization
st.subheader("3D Visualization")
feature1 = st.selectbox("Feature 1", features_to_plot)
feature2 = st.selectbox("Feature 2", features_to_plot)
feature3 = st.selectbox("Feature 3", features_to_plot)

fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[y == 1][feature1], X[y == 1][feature2], X[y == 1][feature3], c='r', marker='o', label='Spam')
ax.scatter(X[y == 0][feature1], X[y == 0][feature2], X[y == 0][feature3], c='g', marker='^', label='Not Spam')
ax.set_xlabel(feature1)
ax.set_ylabel(feature2)
ax.set_zlabel(feature3)
ax.set_title('Spambase Dataset - 3D Visualization')
ax.legend()
st.pyplot(fig)

# Decision Boundaries
st.subheader("Decision Boundaries")
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for i, (name, model) in enumerate(models.items()):
    plot_decision_regions(X_train_2_scaled, y_train.values, clf=model, legend=2, ax=axes[i])
    axes[i].set_title(name)
    axes[i].set_xlabel(top_2_features[0])
    axes[i].set_ylabel(top_2_features[1])
st.pyplot(fig)

# ROC Curves and AUC
st.subheader("ROC Curves and AUC")
y_prob_knn_2 = models["KNN"].predict_proba(X_test_2_scaled)[:, 1]
y_prob_log_reg_2 = models["Logistic Regression"].predict_proba(X_test_2_scaled)[:, 1]
y_prob_dt_2 = models["Decision Tree"].predict_proba(X_test_2_scaled)[:, 1]

roc_auc_knn_test = roc_auc_score(y_test, y_prob_knn_2)
roc_auc_log_reg_test = roc_auc_score(y_test, y_prob_log_reg_2)
roc_auc_dt_test = roc_auc_score(y_test, y_prob_dt_2)

fpr_knn_2, tpr_knn_2, _ = roc_curve(y_test, y_prob_knn_2)
fpr_log_reg_2, tpr_log_reg_2, _ = roc_curve(y_test, y_prob_log_reg_2)
fpr_dt_2, tpr_dt_2, _ = roc_curve(y_test, y_prob_dt_2)

fig, ax = plt.subplots(figsize=(10, 8))
ax.plot(fpr_knn_2, tpr_knn_2, color='blue', lw=2, label=f'KNN (AUC = {roc_auc_knn_test:.2f})')
ax.plot(fpr_log_reg_2, tpr_log_reg_2, color='green', lw=2, label=f'Logistic Regression (AUC = {roc_auc_log_reg_test:.2f})')
ax.plot(fpr_dt_2, tpr_dt_2, color='red', lw=2, label=f'Decision Tree (AUC = {roc_auc_dt_test:.2f})')

ax.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=2)

ax.set_title('ROC Curves for Classifiers', fontsize=16)
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)

ax.legend(loc='lower right', fontsize=12)

ax.grid(True)
st.pyplot(fig)

# AUC Table
data = {
    'Classifier': ['KNN', 'Logistic Regression', 'Decision Tree'],
    'AUC (test)': [roc_auc_knn_test, roc_auc_log_reg_test, roc_auc_dt_test]
}
auc_df = pd.DataFrame(data)
st.dataframe(auc_df)

# Input Features (Sidebar)
with st.sidebar:
    st.header('Input Features')
    input_data = {}
    for feature in all_features: # <- Используем all_features здесь
        if data[feature].dtype == 'int64' or data[feature].dtype == 'float64':
            if feature in data.columns: # <- Проверяем, есть ли признак в data
                min_val = data[feature].min()
                max_val = data[feature].max()
                input_data[feature] = st.slider(feature, min_val, max_val, data[feature].mean())
            else: # <- Если признака нет в data, используем значения по умолчанию
                input_data[feature] = st.slider(feature, 0, 100, 50) # или другие значения по умолчанию
        else:
            if feature in data.columns: # <- Проверяем, есть ли признак в data
                unique_vals = data[feature].unique()
                input_data[feature] = st.selectbox(feature, unique_vals)
            else: # <- Если признака нет в data, устанавливаем значение по умолчанию
                input_data[feature] = st.selectbox(feature, ["value1", "value2"]) # или другие значения по умолчанию

    if st.button('Predict'):
        input_df = pd.DataFrame([input_data])
        # Масштабирование входных данных
        input_scaled = scaler.transform(input_df[features])

        # Предсказание
        best_model_name = auc_df.loc[auc_df['AUC (test)'].idxmax(), 'Classifier']  # Получаем имя лучшей модели
        best_model = models[best_model_name]  # Получаем лучшую модель
        prediction = best_model.predict(input_scaled)
        prediction_proba = best_model.predict_proba(input_scaled)

        # Display Prediction
        st.subheader('Predicted Spam')
        st.success(f"Predicted class: **{prediction[0]}**")

        # Display Probabilities
        proba_df = pd.DataFrame(prediction_proba, columns=best_model.classes_)
        st.dataframe(
            proba_df,
            column_config={
                col: st.column_config.ProgressColumn(col, format='%f', min_value=0, max_value=1)
                for col in proba_df.columns
            },
            hide_index=True
        )
