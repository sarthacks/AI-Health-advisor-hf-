# Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

# Corrected file path
DATA_PATH = "illnessnames/file2.csv"

# Reading the dataset
data = pd.read_csv(DATA_PATH).dropna(axis=1)

# Encoding the target value
encoder = LabelEncoder()
data["prognosis"] = encoder.fit_transform(data["prognosis"])

# Splitting the data
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Training models on the whole dataset
final_svm_model = SVC()
final_nb_model = GaussianNB()
final_rf_model = RandomForestClassifier(random_state=18)

final_svm_model.fit(X, y)
final_nb_model.fit(X, y)
final_rf_model.fit(X, y)

# Creating a symptom index dictionary
symptoms = X.columns.values
symptom_index = {symptom.replace("_", " ").capitalize(): i for i, symptom in enumerate(symptoms)}

data_dict = {
    "symptom_index": symptom_index,
    "predictions_classes": encoder.classes_
}

# Function to predict disease from symptoms
def predict_disease(symptoms_input):
    symptoms_list = [s.strip().capitalize() for s in symptoms_input.split(",") if s.strip()]
    
    # Handling empty input
    if not symptoms_list:
        print("No symptoms entered. Please provide at least two symptoms.")
        return
    
    # Handling single input case
    if len(symptoms_list) < 2:
        print("Please enter at least two symptoms for better accuracy.")
        return
    
    # Creating input vector for the model
    input_data = [0] * len(data_dict["symptom_index"])
    for symptom in symptoms_list:
        index = data_dict["symptom_index"].get(symptom, None)
        if index is not None:
            input_data[index] = 1
    
    input_data = pd.DataFrame([input_data], columns=X.columns)
    
    # Generating predictions
    rf_prediction = data_dict["predictions_classes"][final_rf_model.predict(input_data)[0]]
    nb_prediction = data_dict["predictions_classes"][final_nb_model.predict(input_data)[0]]
    svm_prediction = data_dict["predictions_classes"][final_svm_model.predict(input_data)[0]]
    
    # Final prediction based on majority voting
    final_prediction = max(set([rf_prediction, nb_prediction, svm_prediction]), key=[rf_prediction, nb_prediction, svm_prediction].count)
    
    # Printing results
    print("Predicted Disease Information:")
    print(f"Random Forest Prediction: {rf_prediction}")
    print(f"Naive Bayes Prediction: {nb_prediction}")
    print(f"SVM Prediction: {svm_prediction}")
    print(f"Final Prediction (Majority Vote): {final_prediction}")

# Taking user input for symptoms
while True:
    user_input = input("Enter symptoms separated by commas: ").strip()
    if user_input:  # Ensuring input is not empty
        predict_disease(user_input)
        break
    else:
        print("No input detected. Please enter your symptoms.")


