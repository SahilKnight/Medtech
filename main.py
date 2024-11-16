from flask import Flask, request, render_template, jsonify
import numpy as np
import pandas as pd
import pickle

app = Flask(__name__)

# Load datasets
sym_des = pd.read_csv("datasets/symtoms_df.csv")
precautions = pd.read_csv("datasets/precautions_df.csv")
workout = pd.read_csv("datasets/workout_df.csv")
description = pd.read_csv("datasets/description.csv")
medications = pd.read_csv('datasets/medications.csv')
diets = pd.read_csv("datasets/diets.csv")

# Load model
svc = pickle.load(open('models/svc.pkl', 'rb'))

# Helper function
def helper(dis):
    desc = " ".join(description[description['Disease'] == dis]['Description'].values)
    pre = precautions[precautions['Disease'] == dis][['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']].values.flatten()
    med = medications[medications['Disease'] == dis]['Medication'].values
    die = diets[diets['Disease'] == dis]['Diet'].values
    wrkout = workout[workout['disease'] == dis]['workout'].values
    return desc, pre, med, die, wrkout

# Symptoms and disease dictionaries
symptoms_dict = {
    'itching': 0, 'skin_rash': 1, 'nodal_skin_eruptions': 2, 'continuous_sneezing': 3, 'shivering': 4, 
    # Add additional symptoms here
}
diseases_list = {
    15: 'Fungal infection', 4: 'Allergy', 16: 'GERD', 9: 'Chronic cholestasis',
    # Add additional diseases here
}

# Prediction function
def get_predicted_value(patient_symptoms):
    input_vector = np.zeros(len(symptoms_dict))
    for symptom in patient_symptoms:
        if symptom in symptoms_dict:
            input_vector[symptoms_dict[symptom]] = 1
    return diseases_list[svc.predict([input_vector])[0]]

@app.route("/")
def index():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    symptoms = request.form.getlist('symptoms')
    if not symptoms:
        message = "Please select symptoms for a prediction."
        return render_template('index.html', message=message)

    predicted_disease = get_predicted_value(symptoms)
    dis_des, precautions, medications, rec_diet, workout = helper(predicted_disease)

    return render_template('index.html', predicted_disease=predicted_disease, dis_des=dis_des,
                           my_precautions=precautions, medications=medications, my_diet=rec_diet,
                           workout=workout)

@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/contact')
def contact():
    return render_template("contact.html")

@app.route('/developer')
def developer():
    return render_template("developer.html")

@app.route('/blog')
def blog():
    return render_template("blog.html")

@app.route('/book-appointment')
def book_appointment():
    return render_template('appointment.html')

@app.route('/upload-report')
def upload_report():
    return render_template('UploadReport.html')

if __name__ == '__main__':
    app.run(debug=True)
