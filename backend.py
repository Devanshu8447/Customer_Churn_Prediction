import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

# Load scaler and model
scaler = joblib.load('scaler.joblib')
model = load_model('churn_model.h5')

MODEL_FEATURES = [
    'CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard',
    'IsActiveMember', 'EstimatedSalary', 'Geography_Germany', 'Geography_Spain', 'Gender_Male'
]

def preprocess_input(input_dict):
    df = pd.DataFrame([input_dict])
    
    # One hot encode Geography and Gender manually
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

def predict_churn(input_dict):
    processed = preprocess_input(input_dict)
    prediction_prob = model.predict(processed)[0][0]
    prediction = prediction_prob >= 0.5
    return {
        'churn_probability': float(prediction_prob),
        'churn': bool(prediction)
    }
