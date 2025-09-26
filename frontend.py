import streamlit as st
from backend import predict_churn

st.title("Customer Churn Prediction")

credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=650)
age = st.number_input("Age", min_value=18, max_value=100, value=40)
tenure = st.number_input("Tenure (years)", min_value=0, max_value=10, value=3)
balance = st.number_input(
    "Account Balance", min_value=0.0, max_value=250000.0, value=50000.0
)
num_of_products = st.selectbox("Number of Products", options=[1, 2, 3, 4], index=0)
has_cr_card = st.selectbox("Has Credit Card?", options=["No", "Yes"], index=1)
is_active_member = st.selectbox("Is Active Member?", options=["No", "Yes"], index=1)
estimated_salary = st.number_input(
    "Estimated Salary", min_value=10000.0, max_value=200000.0, value=50000.0
)
geography = st.selectbox("Geography", options=["France", "Germany", "Spain"], index=0)
gender = st.selectbox("Gender", options=["Female", "Male"], index=0)

if st.button("Predict Churn"):
    input_data = {
        "CreditScore": credit_score,
        "Age": age,
        "Tenure": tenure,
        "Balance": balance,
        "NumOfProducts": num_of_products,
        "HasCrCard": 1 if has_cr_card == "Yes" else 0,
        "IsActiveMember": 1 if is_active_member == "Yes" else 0,
        "EstimatedSalary": estimated_salary,
        "Geography": geography,
        "Gender": gender,
    }
    result = predict_churn(input_data)
    st.write(f"Churn Probability: {result['churn_probability']:.2%}")
    if result["churn"]:
        st.error("Customer is likely to churn.")
    else:
        st.success("Customer is unlikely to churn.")
