from flask import Flask, render_template, request
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

app = Flask(__name__)

# Load the pre-trained model and scaler
model = joblib.load('telco_churn_model.pkl')
scaler = joblib.load('telco_churn_scaler.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get user input from the form
        tenure = int(request.form['tenure'])
        monthly_charges = float(request.form['monthly_charges'])
        total_charges = float(request.form['total_charges'])
        contract = int(request.form['contract'])
        online_security = int(request.form['online_security'])
        tech_support = int(request.form['tech_support'])
        
        # Create a DataFrame with user input
        data = {'tenure': [tenure],
                'MonthlyCharges': [monthly_charges],
                'TotalCharges': [total_charges],
                'Contract': [contract],
                'OnlineSecurity': [online_security],
                'TechSupport': [tech_support]}
        
        user_input = pd.DataFrame(data)
        
        # Scale the user input
        scaled_input = scaler.transform(user_input)
        
        # Make prediction
        prediction = model.predict(scaled_input)[0]
        
        # Display the prediction on the result page
        return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
