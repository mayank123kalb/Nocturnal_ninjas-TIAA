import numpy as np
import pandas as pd
from flask import Flask, request, redirect, render_template, jsonify
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib
import google.generativeai as palm


app = Flask(__name__)

API_KEY = 'AIzaSyA4Q2RuZuU5p65TOupateJOatfm_jYYXqA'
palm.configure(api_key=API_KEY)

model = joblib.load('investment_strategy_model.pkl')

# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Load the training data to extract feature names
df = pd.read_excel('investment_dataset_strategies.xlsx')
X_encoded = pd.get_dummies(df.drop('Investment Strategy', axis=1), columns=['Financial Goals'])

# Fit LabelEncoder on the target variable
label_encoder.fit(df['Investment Strategy'])

# Extract feature names
feature_names = X_encoded.columns


@app.route('/',methods=["GET", "POST"])
def index():
  return render_template('index.html')

@app.route('/finance')
def finance():
    return render_template('finance.html')

@app.route('/diet')
def diet():
    return render_template('diet.html')

a = 0
@app.route('/ask', methods=['POST'])
def ask():
    global a
    user_message = request.form['user_message']
    response = palm.chat(messages=user_message, temperature=0.1, context=' Speak like a certified trainer and dietician,take inputs such as ask him about age,diet, and other relevent features to sugesst him diet as per he needs')
    chat_response = response.last

    if(a==0):
        chat_response = "I can suggest you diet , exercises and also count your calorie intakes what will you like to do ??"
        a=1
    return jsonify({'response': chat_response})


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get user input from the form
        age = int(request.form['age'])
        income = int(request.form['income'])
        financial_goals = request.form['financial_goals']
        investment_amount = int(request.form['investment_amount'])

        # Convert user input into a DataFrame
        user_input = pd.DataFrame({
            'Age': [age],
            'Income': [income],
            'Investment Amount': [investment_amount],
            'Financial Goals_' + financial_goals: [1]  # One-hot encode the Financial Goals
        })

        # Add missing columns in user_input that are present in feature_names
        missing_columns = set(feature_names) - set(user_input.columns)
        for column in missing_columns:
            user_input[column] = 0

        # Reorder columns to match the model's feature order
        user_input = user_input[feature_names]

        # Make a prediction using the loaded model
        prediction = model.predict(user_input)

        # Inverse transform the numeric prediction back to the original class
        predicted_class = label_encoder.inverse_transform(prediction)

        # Render the result on the HTML page
        return render_template('finance.html', prediction=f"The predicted investment strategy is: {predicted_class[0]}")



    except Exception as e:
        return render_template('finance.html', error=f"An error occurred: {str(e)}")



if __name__ == "__main__":
    app.run(debug=False)