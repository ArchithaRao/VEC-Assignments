# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 17:52:23 2024

@author: Architha Rao
"""
from flask import Flask, render_template, request
from sklearn.preprocessing import LabelEncoder
import pickle
import os

app = Flask(__name__)

# Load models and encoders
try:
    model_path = os.path.join(os.path.dirname(__file__), 'ar_xgb.pkl')
    ss_path = os.path.join(os.path.dirname(__file__), 'ar_ss.pkl')

    with open(model_path, 'rb') as model_file:
        model = pickle.load(model_file)
    with open(ss_path, 'rb') as ss_file:
        ss1 = pickle.load(ss_file)

    le1 = LabelEncoder()
    le2 = LabelEncoder()
    le3 = LabelEncoder()
    le4 = LabelEncoder()
    le5 = LabelEncoder()
    le6 = LabelEncoder()
    le7 = LabelEncoder()
    le8 = LabelEncoder()
    le9 = LabelEncoder()
    le10 = LabelEncoder()

    # Assuming we need to fit the encoders here
    # Example: le1.fit(['Airline1', 'Airline2', ...])
    # Repeat for other encoders

except Exception as e:
    print(f"Error loading models or encoders: {e}")

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/predict')
def predict():
    return render_template("predict.html")

@app.route('/submit', methods=['GET', 'POST'])
def submit():
    if request.method == 'POST':
        try:
            # Retrieve form data
            Airline_name = request.form['Airline name']
            Seat_Type = request.form['Seat Type']
            Type_of_traveller = request.form['Type Of Traveller']
            Origin = request.form['Origin']
            Destination = request.form['Destination']
            Month_Flown = request.form['Month Flown']
            Year_Flown = request.form['Year Flown']
            Verified = request.form['Verified']
            S_C = request.form['S_C']
            F_B = request.form['F_B']
            G_S = request.form['G_S']
            O_R = request.form['O_R']

            # Encode categorical data
            encoded_data = [
                le1.transform([Airline_name])[0],
                le2.transform([Seat_Type])[0],
                le3.transform([Type_of_traveller])[0],
                le4.transform([Origin])[0],
                le5.transform([Destination])[0],
                le6.transform([Month_Flown])[0],
                le7.transform([Year_Flown])[0],
                le8.transform([Verified])[0],
                int(S_C),  # Convert to int
                int(F_B),  # Convert to int
                int(G_S),  # Convert to int
                le9.transform([O_R])[0]
            ]

            # Print encoded data for debugging
            print(encoded_data)

            # Make prediction
            prediction = model.predict(ss1.transform([encoded_data]))

            # Determine recommendation
            recommendation = "Recommended" if prediction == 1 else "Not Recommended"

        except Exception as e:
            print(f"Error during form processing or prediction: {e}")
            return render_template('submit.html', error=str(e))

    # Render submit page initially or on GET request
    return render_template('submit.html')

if __name__ == '__main__':
    app.run(debug=False, port=5000)
