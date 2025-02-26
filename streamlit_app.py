import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from mlxtend.plotting import plot_decision_regions
from sklearn.metrics import roc_curve, roc_auc_score
import warnings
warnings.filterwarnings("ignore")

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

print(len(column_names))
column_names

data = pd.read_csv("spambase.data", delimiter=',', header=None, names=column_names)
print(data.sample(10, random_state=42))
print(data.nunique())
print(data.shape)
print(data.describe())

data = data.dropna(subset=["spam"])
print(data)

unique_classes = data['spam'].unique()
if len(unique_classes) > 2:
    print("Исходное распределение классов:")
    print(data['spam'].value_counts())
    threshold = np.median(data['spam'])
    data['binary_label'] = (data['spam'] > threshold).astype(int)
    print("\nРаспределение после объединения в бинарную классификацию:")
    print(data['binary_label'].value_counts())
else:
    print("Бинарная классификация уже присутствует.")

major_class = data['spam'].value_counts().idxmax()
data['spam'] = (data['spam'] == major_class).astype(int)
print(data['spam'].value_counts())

numeric_columns = data.columns[data.dtypes == 'object']
for col in numeric_columns:
    data[col] = pd.to_numeric(data[col], errors='coerce')
print(data.dtypes)

data = data.select_dtypes(include=['number'])
print(data.dtypes)

data_pos = data[data["spam"] == 1]
data_neg = data[data["spam"] == 0]
data_pos = data_pos.fillna(data_pos.mean())
data_neg = data_neg.fillna(data_neg.mean())
data = pd.concat([data_pos, data_neg]).sort_index()
print(data.isnull().sum().sum())

target = 'spam'
features = [i for i in data.columns if i != target]
X = data[features]
y = data[target]

features_up_10_unique = X.loc[:, X.nunique() > 10]
correlations = features_up_10_unique.corrwith(y).abs()
top_features = correlations.nlargest(3)
print(top_features)

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
plt.show()

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

plt.figure(figsize=(15, 5))
plt.title('ROC Curves for Classifiers')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.grid(True)
plt.show()

y_prob_log_reg_test = log_reg_2.predict_proba(X_test_2_scaled)[:, 1]
y_prob_dt_test = decision_tree_2.predict_proba(X_test_2_scaled)[:, 1]

roc_auc_log_reg_test = roc_auc_score(y_test, y_prob_log_reg_test)
roc_auc_dt_test = roc_auc_score(y_test, y_prob_dt_test)

data = {
    'Classifier': ['Logistic Regression', 'Decision Tree'],
    'AUC (test)': [roc_auc_log_reg_test, roc_auc_dt_test]
}
auc_df = pd.DataFrame(data)
print(auc_df)
