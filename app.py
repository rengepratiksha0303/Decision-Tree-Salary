import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# App title
st.title("💼 Salary Prediction using Decision Tree")

st.write("This app predicts salary based on input features using a Decision Tree Regression model.")

# Load dataset
@st.cache_data
def load_data():
    data = pd.read_csv("Salary.csv")
    return data

data = load_data()

st.subheader("📊 Dataset Preview")
st.dataframe(data.head())

# Split features and target
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = DecisionTreeRegressor(max_depth=5, random_state=42)
model.fit(X_train, y_train)

# Model performance
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)

st.subheader("📈 Model Performance")
st.write(f"**R² Score:** {r2:.2f}")

# User input section
st.subheader("🧮 Enter Input Values")

user_input = []
for col in X.columns:
    val = st.number_input(f"Enter {col}", float(X[col].min()), float(X[col].max()))
    user_input.append(val)

# Prediction
if st.button("Predict Salary"):
    prediction = model.predict([user_input])
    st.success(f"💰 Predicted Salary: {prediction[0]:,.2f}")
