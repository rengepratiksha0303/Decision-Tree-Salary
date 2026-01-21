import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

st.set_page_config(page_title="Salary Prediction", layout="centered")

st.title("💼 Salary Prediction using Decision Tree")

st.write("Upload your **Salary.csv** file to train the model.")

# ---------------- FILE UPLOAD ----------------
uploaded_file = st.file_uploader("Upload Salary CSV file", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    st.subheader("📊 Dataset Preview")
    st.dataframe(data.head())

    # Features & Target
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model
    model = DecisionTreeRegressor(max_depth=5, random_state=42)
    model.fit(X_train, y_train)

    # Model evaluation
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)

    st.subheader("📈 Model Performance")
    st.write(f"**R² Score:** {r2:.2f}")

    # ---------------- USER INPUT ----------------
    st.subheader("🧮 Enter Input Values")

    user_input = []
    for col in X.columns:
        val = st.number_input(
            f"{col}",
            float(X[col].min()),
            float(X[col].max())
        )
        user_input.append(val)

    if st.button("Predict Salary"):
        prediction = model.predict([user_input])
        st.success(f"💰 Predicted Salary: {prediction[0]:,.2f}")

else:
    st.warning("⚠️ Please upload a CSV file to continue.")
