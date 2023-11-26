import numpy as np
import pandas as pd
from flask import Flask, request, redirect, render_template, jsonify
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib
import google.generativeai as palm

import tensorflow as tf
from flask import Flask, request, render_template
from tensorflow.keras.preprocessing.image import load_img, img_to_array

app = Flask(__name__)

API_KEY = 'AIzaSyA4Q2RuZuU5p65TOupateJOatfm_jYYXqA'
palm.configure(api_key=API_KEY)

model = joblib.load('investment_strategy_model.pkl')

existing_model = tf.keras.models.load_model('aaaaa.hdf5', compile = False)


class_names = ["Diversify across different asset classes for a balanced approach.: "
, "Prioritize long-term growth with a significant allocation to equity investments."
, "Focus on capital preservation with conservative investment choices."
, "Utilize tax-advantaged retirement accounts like a provident fund or a pension plan."]

info = {
    class_names[0]: {
        "Description": "Diversification involves spreading your investments across various types of assets, such as stocks, bonds, and real estate. This strategy aims to reduce risk by avoiding over-reliance on a single type of investment. A balanced approach considers the long-term benefits of having a "
                       "mix of assets that may perform differently under various market conditions."
    },
    class_names[1]: {
        "Description": "Emphasizing long-term growth involves allocating a substantial portion of your portfolio to equities (stocks) with the expectation that, over time, the value of these investments will increase. This strategy often requires patience and the ability to weather short-term market fluctuations, as the focus is "
                       "on the potential for significant appreciation in the value of the investments."
    },
    class_names[2]: {
        "Description": "A conservative investment strategy prioritizes the preservation of capital over high returns. Investors following this approach typically choose lower-risk investments, such as government bonds or blue-chip stocks. While the potential for high returns may be lower compared to more aggressive strategies,"
                       " the goal is to protect the invested capital from significant losses."
    },
    class_names[3]: {
        "Description": "This strategy involves taking advantage of tax-advantaged retirement accounts to optimize your retirement savings. Examples include provident funds, pension plans, or individual retirement accounts (IRAs). Contributions to these accounts may offer tax benefits, and the funds are often earmarked for retirement, "
                       "providing a structured approach to long-term financial planning."
    }
}


class_namess = ["Acne", "Eczema", "Melanoma", "Normal", "Psoriasis", "Tinea", "vitiligo"]

disease_info = {
    "Acne": {
        "Description": "Acne is a common skin condition that occurs when hair follicles are clogged with oil and dead skin cells. It often causes pimples and can vary in severity.",
        "Causes": "Causes of acne include hormonal changes, excess oil production, bacteria, and certain medications.",
        "Treatment": "Treatment options for acne may include topical creams, oral medications, and lifestyle changes.",
    },
    "Ezcema": {
        "Description": "Acne is a common skin condition that occurs when hair follicles are clogged with oil and dead skin cells. It often causes pimples and can vary in severity.",
        "Causes": "Causes of acne include hormonal changes, excess oil production, bacteria, and certain medications.",
        "Treatment": "Treatment options for acne may include topical creams, oral medications, and lifestyle changes.",
    },
    "Melanoma": {
        "Description": "Melanoma is a type of skin cancer that originates in the melanocytes, the pigment-producing cells in the skin. It is considered the most dangerous form of skin cancer because it can rapidly spread to other parts of the body if not detected and treated early. Melanoma often appears as an unusual mole or pigmented skin lesion that can change in size, shape, or color over time.",
        "Causes": "The primary cause of melanoma is exposure to ultraviolet (UV) radiation from sunlight or artificial sources like tanning beds. Individuals with a history of excessive sun exposure, sunburns, a family history of melanoma, numerous moles, fair skin, and a weakened immune system are at higher risk. UV radiation damages DNA in skin cells, leading to the development of cancerous cells.",
        "Treatment": "Treatment options for melanoma depend on the stage of the cancer and may include:Surgery, Immunotherapy, Targeted Therapy, Chemotherapy, Radiation Therapy, Clinical Trials",
    },
    "Normal":{
        "Description": "Normal",
        "Causes": "Normal",
        "Treatment": "Normal",
    },
    "Psoriasis":{
        "Description": "Psoriasis is a chronic autoimmune skin condition characterized by the rapid buildup of skin cells, leading to the formation of red, scaly patches on the skin's surface. It can affect various parts of the body and often has periods of exacerbation and remission.",
        "Causes": "The exact cause of psoriasis is not fully understood, but it is thought to involve a combination of genetic, immune system, and environmental factors. Triggers can include stress, infections, certain medications, and injuries to the skin.",
        "Treatment": "Treatment for psoriasis aims to reduce inflammation, slow down skin cell growth, and alleviate symptoms. Common treatments include topical corticosteroids, phototherapy, oral medications, and biologic drugs. Lifestyle modifications, such as managing stress and avoiding triggers, can also help manage the condition.",
    },
    "Tinea":{
        "Description": "Tinea, commonly known as ringworm, is a contagious fungal infection that affects the skin, scalp, or nails. It often presents as a red, itchy rash with a ring-like appearance. Despite its name, ringworm is not caused by a worm but by various types of fungi.",
        "Causes": "Tinea is caused by different species of fungi known as dermatophytes. These fungi thrive in warm and humid environments and can be transmitted through direct contact with an infected person or contaminated objects.",
        "Treatment": "Treatment for tinea typically involves antifungal medications, either topical (creams or ointments) or oral (pills). Good hygiene practices, such as keeping the affected area clean and dry, are important for preventing the spread of the infection.",
    },
    "vitiligo":{
        "Description": "Vitiligo is a long-term skin condition characterized by the loss of skin pigment, resulting in white patches on the skin. It occurs when melanocytes, the cells responsible for producing pigment, are destroyed or stop functioning.",
        "Causes": "The exact cause of vitiligo is not fully understood, but it is believed to involve a combination of genetic and autoimmune factors. It is not contagious, and it can occur at any age.",
        "Treatment": "While there is no cure for vitiligo, various treatments can help manage the condition. These treatments may include topical corticosteroids, topical calcineurin inhibitors, narrowband ultraviolet B (NB-UVB) therapy, and surgical options such as skin grafting. Treatment choices depend on the extent and location of the white patches and should be discussed with a dermatologist.",
    },
}

disease_videos = {
    "Acne": "https://youtu.be/phoCNpzyNG4?si=Uk-sP3S_x0LcOBbj",
    "Eczema": "https://youtu.be/f_sMpdifzVc?si=8wZaVbjtm6ibrOcL",
    "Melanoma": "https://youtu.be/ZwM5EZYuV1o?si=LfqhSJAhGjm2BTyA",
    "Normal": "https://www.youtube.com/watch?v=VIDEO_ID_FOR_NORMAL",
    "Psoriasis": "https://youtu.be/MbmizU2O1XY?si=5V6Or98U0NOYN9Ty",
    "Tinea": "https://youtu.be/GpG22UKhMNw?si=Y_mz0H-K3zbake17",
    "vitiligo": "https://youtu.be/Zz35mjTdse4?si=Fc4BFYoDTRH22gmc",
}





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

@app.route('/financehome')
def financehome():
    return render_template('financehome.html')

@app.route('/diet')
def diet():
    return render_template('diet.html')

@app.route('/healthhome')
def healthhome():
    return render_template('healthhome.html')

@app.route('/consultation')
def consultation():
    return render_template('consultation.html')


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

        if predicted_class[0] in info:
            predicted_description = info[predicted_class[0]]["Description"]
        # Render the result on the HTML page with the predicted class
        return render_template('finance.html', prediction=f"The predicted investment strategy is: {predicted_class[0]}", description=predicted_description)

    except Exception as e:
        # Render the error on the HTML page
        return render_template('finance.html', error=f"An error occurred: {str(e)}")

def predict_skin_disease(image_path):
    img = load_img(image_path, target_size=(224, 224))  # Adjust target_size as needed
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.efficientnet_v2.preprocess_input(img_array)
    predictions = existing_model.predict(img_array)
    predicted_class_index = np.argmax(predictions)
    predicted_class_name = class_namess[predicted_class_index]
    return predicted_class_name

@app.route('/common', methods=['GET', 'POST'])
def common():
    predicted_class = None
    disease_description = None
    disease_causes = None
    disease_treatment = None
    disease_video_url = None
    if request.method == 'POST':
        # Check if a file was uploaded
        if 'file' not in request.files:
            return render_template('index.html', error='No file part')
        file = request.files['file']
        # Check if the file has a filename
        if file.filename == '':
            return render_template('index.html', error='No selected file')
        if file:
            # Save the uploaded file to a temporary location
            temp_file_path = "temp.jpg"
            file.save(temp_file_path)
            # Get the predicted class
            predicted_class = predict_skin_disease(temp_file_path)
            # Fetch disease information and video URL based on the predicted class
            if predicted_class in disease_info:
                disease_description = disease_info[predicted_class]["Description"]
                disease_causes = disease_info[predicted_class]["Causes"]
                disease_treatment = disease_info[predicted_class]["Treatment"]
                if predicted_class in disease_videos:
                    disease_video_url = disease_videos[predicted_class]
    return render_template('common.html', predicted_class=predicted_class, disease_description=disease_description,
                           disease_causes=disease_causes, disease_treatment=disease_treatment,
                           disease_video_url=disease_video_url)


if __name__ == "__main__":
    app.run(debug=False)