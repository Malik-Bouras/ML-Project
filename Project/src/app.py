import os
import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template

app = Flask(__name__,template_folder='templates')

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    # loading the saved model and scaler
    model_output_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '.', '..', '..', "Project", "src", "models"))
    scaler = joblib.load(os.path.join(model_output_directory, 'scaler.joblib'))
    model = joblib.load(os.path.join(model_output_directory, 'model.joblib'))
    
    # separating numeric and categorical features
    numeric_fields = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking']
    categorical_fields = ['mainroad', 'guestroom', 'basement', 'airconditioning', 'prefarea']

    # extracting data from form
    form_data = {}
    for field in numeric_fields:
        form_data[field] = int(request.form[field])
    for field in categorical_fields:
        form_data[field] = 1 if (request.form[field] == 'yes') else 0

    # treating this column separately because it has 3 values different from yes/no
    form_data['furnishingstatus'] = 0 if (request.form['furnishingstatus'] == 'furnished') else 1 if (request.form['furnishingstatus'] == 'semi furnished') else 2

    # converting form data to dataframe
    columns = ['area', 'bedrooms', 'bathrooms', 'stories', 'mainroad', 'guestroom', 'basement', 'airconditioning', 'parking', 'prefarea', 'furnishingstatus']
    form_df = pd.DataFrame([form_data])[columns]

    # scaling form data
    scaled_form_data = scaler.transform(form_df)

    # using the model to make a prediction
    prediction = model.predict(scaled_form_data)

    # rounding the prediction to 2 decimal places
    result = round(prediction[0], 2)

    # rendering the index.html template with the predicted price displayed
    return render_template('index.html', prediction_text='The house price should be $ {}'.format(result))


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=80,debug=True)