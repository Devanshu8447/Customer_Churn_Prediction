from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load scaler and model
scaler = joblib.load('scaler.joblib')
model = load_model('churn_model.h5')

MODEL_FEATURES = [
    'CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard',
    'IsActiveMember', 'EstimatedSalary', 'Geography_Germany', 'Geography_Spain', 'Gender_Male'
]

def preprocess_input(input_dict):
    df = pd.DataFrame([input_dict])
    df['Geography_Germany'] = 0
    df['Geography_Spain'] = 0
    if df.loc[0, 'Geography'] == 'Germany':
        df.loc[0, 'Geography_Germany'] = 1
    elif df.loc[0, 'Geography'] == 'Spain':
        df.loc[0, 'Geography_Spain'] = 1
    df['Gender_Male'] = 1 if df.loc[0, 'Gender'] == 'Male' else 0
    df = df.drop(columns=['Geography', 'Gender'])
    df = df[MODEL_FEATURES]
    features_scaled = scaler.transform(df)
    return features_scaled

@app.route('/', methods=['GET', 'POST'])
def home():
    result = None
    churn_prob = None
    if request.method == 'POST':
        input_data = {
            'CreditScore': float(request.form['credit_score']),
            'Age': float(request.form['age']),
            'Tenure': float(request.form['tenure']),
            'Balance': float(request.form['balance']),
            'NumOfProducts': int(request.form['num_of_products']),
            'HasCrCard': int(request.form.get('has_cr_card', 0)),
            'IsActiveMember': int(request.form.get('is_active_member', 0)),
            'EstimatedSalary': float(request.form['estimated_salary']),
            'Geography': request.form['geography'],
            'Gender': request.form['gender']
        }

        processed = preprocess_input(input_data)
        prediction_prob = model.predict(processed)[0][0]
        churn_prob = f"{prediction_prob:.2%}"
        result = prediction_prob >= 0.5

    return render_template('index.html', result=result, churn_prob=churn_prob)

if __name__ == '__main__':
    app.run(debug=True)
