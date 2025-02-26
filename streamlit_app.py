import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

st.title('🎈 My Name Is No One')

st.write('Hello no where!')

# Load the penguins dataset
df = pd.read_csv("https://raw.githubusercontent.com/dataprofessor/data/master/penguins_cleaned.csv")

with st.expander('Data'):
    st.write("X")
    X_raw = df.drop('species', axis=1)
    st.dataframe(X_raw)

    st.write('y')
    y_raw = df.species
    st.dataframe(y_raw)

# Sidebar input
with st.sidebar:
    st.header('Enter Penguin Features: ')
    island = st.selectbox('Island', ('Torgersen', 'Dream', 'Biscoe'))
    bill_length_mm = st.slider('Bill length (mm)', 32.1, 59.6, 44.5)
    bill_depth_mm = st.slider('Bill depth (mm)', 13.1, 21.5, 17.3)
    flipper_length_mm = st.slider('Flipper length (mm)', 32.1, 59.6, 44.5)
    body_mass_g = st.slider('Body mass (g)', 32.1, 59.6, 44.5)
    gender = st.selectbox('Gender', ("female", 'male'))

# Data visualization
st.subheader('Data Visualization')

# Scatter plot for Bill Length vs Bill Depth
fig = px.scatter(
    df,
    x='bill_length_mm',
    y='bill_depth_mm',
    color='island',
    title='Bill Length vs. Bill Depth by Island'
)
st.plotly_chart(fig)

# Histogram for Body Mass Distribution
fig2 = px.histogram(
    df,
    x='body_mass_g',
    nbins=30,
    title='Distribution of Body Mass'
)
st.plotly_chart(fig2)

# Prepare input data
data = {
    'island': island,
    'bill_length_mm': bill_length_mm,
    'bill_depth_mm': bill_depth_mm,
    'flipper_length_mm': flipper_length_mm,
    'body_mass_g': body_mass_g,
    'sex': gender
}
input_df = pd.DataFrame(data, index=[0])
input_penguins = pd.concat([input_df, X_raw], axis=0)

with st.expander('Input Features'):
    st.write('**Input penguin**')
    st.dataframe(input_df)
    st.write('**Combined penguins data** (input row + original data)')
    st.dataframe(input_penguins)

# Encode categorical features
encode = ['island', 'sex']
df_penguins = pd.get_dummies(input_penguins, prefix=encode)

X = df_penguins[1:]
input_row = df_penguins[:1]

# Target encoding
target_mapper = {'Adelie': 0, 'Chinstrap': 1, 'Gentoo': 2}
def target_encode(val):
    return target_mapper[val]
y = y_raw.apply(target_encode)

with st.expander('Data preparation'):
    st.write('**Encoded X (input penguin)**')
    st.dataframe(input_row)
    st.write(y)

# Model selection via sidebar
model_choice = st.sidebar.selectbox('Choose a model', ['Random Forest', 'Decision Tree', 'KNN'])

# Train the selected model
if model_choice == 'Random Forest':
    model = RandomForestClassifier(random_state=42)
elif model_choice == 'Decision Tree':
    model = DecisionTreeClassifier(random_state=42)
else:
    model = KNeighborsClassifier()

# Training with a progress spinner
with st.spinner(f'Training {model_choice}...'):
    model.fit(X, y)
st.success(f'{model_choice} model trained!')

# Make predictions
prediction = model.predict(input_row)
prediction_proba = model.predict_proba(input_row)
df_prediction_proba = pd.DataFrame(prediction_proba, columns=['Adelie', 'Chinstrap', 'Gentoo'])

# Display prediction probability
st.subheader('Predicted Species')
st.dataframe(
    df_prediction_proba,
    column_config={
        'Adelie': st.column_config.ProgressColumn('Adelie', format='%f', width='medium', min_value=0, max_value=1),
        'Chinstrap': st.column_config.ProgressColumn('Chinstrap', format='%f', width='medium', min_value=0, max_value=1),
        'Gentoo': st.column_config.ProgressColumn('Gentoo', format='%f', width='medium', min_value=0, max_value=1),
    },
    hide_index=True
)

penguins_species = np.array(['Adelie', 'Chinstrap', 'Gentoo'])
st.success(f"Predicted species: **{penguins_species[prediction][0]}**")

# Model performance: Accuracy and confusion matrix
st.subheader('Model Performance')
X_test = df_penguins[1:]
y_test = y_raw.apply(target_encode)
y_pred = model.predict(X_test)

# Accuracy and confusion matrix
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

st.write(f"Accuracy: {acc:.2f}")
st.write('Confusion Matrix:')
st.write(cm)

# Feature Importance (for Random Forest)
if model_choice == 'Random Forest':
    feature_importances = model.feature_importances_
    features = X.columns
    importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)
    fig3 = px.bar(importance_df, x='Feature', y='Importance', title='Feature Importance')
    st.plotly_chart(fig3)

# Display the prediction confidence
st.subheader('Prediction Confidence')
confidences = df_prediction_proba.max(axis=1)
st.write(f"The model's confidence in the prediction: {confidences[0]:.2f}")
st.progress(confidences[0])
