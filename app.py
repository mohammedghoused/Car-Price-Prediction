from flask import Flask, render_template, request
import pandas as pd
import pickle
import numpy as np
import re

app = Flask(__name__)

# Load the cleaned data to populate dropdowns
try:
    car = pd.read_csv("Cleaned Car.csv")
except FileNotFoundError:
    car = pd.DataFrame(columns=['name', 'company', 'year', 'Price', 'kms_driven', 'fuel_type'])

# Load the trained model pipeline
try:
    model = pickle.load(open("CarPriceModel.pkl", 'rb'))
except FileNotFoundError:
    model = None

@app.route('/')
def index():
    if car.empty:
        return "Cleaned Car.csv not found. Please run retrain_model.py first."
    
    companies = sorted(car['company'].unique())
    car_models = sorted(car['name'].unique())
    year = sorted(car['year'].unique(), reverse=True)
    fuel_types = car['fuel_type'].dropna().unique()
    
    # Add a default option
    if "Select Company" not in companies:
        companies.insert(0, "Select Company")
        
    return render_template("index.html", companies=companies, car_models=car_models, years=year, fuel_types=fuel_types)

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return "Model not found. Please train the model first by running retrain_model.py.", 500
        
    try:
        company = request.form.get('company')
        car_model = request.form.get('car_models')
        year_raw = request.form.get('year')
        fuel_type = request.form.get('fuel')
        kilo_raw = request.form.get('kilo')
        
        # Robust parsing of inputs
        year = int(year_raw) if year_raw else 2012
        
        # Clean kilometer input: extract only numbers (e.g. "45,000 kms" -> 45000)
        clean_kilo = re.sub(r'\D', '', kilo_raw) if kilo_raw else '0'
        kms_driven = int(clean_kilo) if clean_kilo else 0
        
        print(f"Prediction requested: Model={car_model}, Company={company}, Year={year}, Kms={kms_driven}, Fuel={fuel_type}")
        
        # Build DataFrame with correct column names matching the training pipeline
        input_data = pd.DataFrame(
            [[car_model, company, year, kms_driven, fuel_type]], 
            columns=['name', 'company', 'year', 'kms_driven', 'fuel_type']
        )
        
        # Make prediction
        prediction = model.predict(input_data)
        
        # Ensure predicted price is not negative (regression can occasionally output negative prices for extreme outliers)
        predicted_price = max(0.0, np.round(prediction[0], 2))
        return str(predicted_price)
        
    except Exception as e:
        print(f"Error during prediction: {e}")
        return f"Prediction Error: {str(e)}", 400

if __name__ == "__main__":
    app.run(debug=True)