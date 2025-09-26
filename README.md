# Customer Churn Prediction Web App

This project is a web application that predicts customer churn using a machine learning model trained on credit card customer data. The app is built with Flask for the backend and Tailwind CSS for a clean responsive frontend UI.

## Features

- Predicts customer churn probability based on customer details
- Uses a trained neural network model with TensorFlow Keras
- Scales input features with saved StandardScaler for accuracy
- Responsive and modern frontend using Tailwind CSS
- Easy-to-use form interface to input customer information
- Clear display of prediction results and probabilities

## Project Structure

- `app.py`: Main Flask backend application managing routes and model inference
- `templates/index.html`: Frontend HTML with Tailwind CSS styling
- `churn_model.h5`: Trained Keras model file (should be placed in project root)
- `scaler.joblib`: Saved StandardScaler object for input scaling
- `.gitignore`: Git ignore rules for typical Python and ML projects
- `requirements.txt`: Python dependencies required to run the app

## Installation

1. Clone the repository:
git clone https://github.com/Devanshu8447/Customer_Churn_Prediction
cd /Customer_Churn_Prediction

2. (Optional) Create a virtual environment and activate it:

python -m venv venv
source venv/bin/activate # On Windows: venv\Scripts\activate

3. Install required packages:
pip install -r requirements.txt

## Usage

Run the Flask app locally:
python app.py

Open your browser and visit `http://127.0.0.1:5000/` to access the web app.

Fill in the customer details form and click **Predict Churn** to get the prediction and churn probability.

## Notes

- Ensure `churn_model.h5` and `scaler.joblib` are in the project root directory before running the app.
- The model expects input features scaled using the saved StandardScaler for consistent predictions.

## Contributing

Contributions are welcome! Please create issues for bugs or feature requests and submit pull requests.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

---
