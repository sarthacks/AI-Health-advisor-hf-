# Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Reading the dataset
DATA_PATH = "illnessnames/file2.csv"
data = pd.read_csv(DATA_PATH).dropna(axis=1)

# Checking if the dataset is balanced
disease_counts = data["prognosis"].value_counts()
temp_df = pd.DataFrame({"Disease": disease_counts.index, "Counts": disease_counts.values})

plt.figure(figsize=(18, 8))
sns.barplot(x="Disease", y="Counts", data=temp_df)
plt.xticks(rotation=90)
plt.show()

# Encoding the target value
encoder = LabelEncoder()
data["prognosis"] = encoder.fit_transform(data["prognosis"])


# Splitting the data
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=24, shuffle=True)


print("Train Columns:", X_train.columns.tolist())
print("Test Columns:", X_test.columns.tolist())

if X_train.columns.tolist() != X_test.columns.tolist():
    print("⚠️ Column mismatch detected! Check data preprocessing steps.")
 

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("Number of common rows:", (X_train.merge(X_test, how='inner').shape[0]))


print(f"Train: {X_train.shape}, {y_train.shape}")
print(f"Test: {X_test.shape}, {y_test.shape}")

# Remove leading/trailing spaces in column names
X_train.columns = X_train.columns.str.strip()
X_test.columns = X_test.columns.str.strip()
train_cols = set(X_train.columns)
test_cols = set(X_test.columns)

extra_train_cols = train_cols - test_cols
extra_test_cols = test_cols - train_cols

print("Extra columns in training data:", extra_train_cols)
print("Extra columns in testing data:", extra_test_cols)

#dropping duplicates
X_train = X_train.drop(columns=['fluid_overload.1'], errors='ignore')
X_test = X_test.drop(columns=['fluid_overload.1'], errors='ignore')

#relign if from diffrent col
X_test = X_test[X_train.columns]

X_train_sorted = X_train.sort_index(axis=1).reset_index(drop=True)
X_test_sorted = X_test.sort_index(axis=1).reset_index(drop=True)


print("X_train columns:", X_train.columns.tolist())
print("X_test columns:", X_test.columns.tolist())
print("\nX_train dtypes:\n", X_train.dtypes)
print("\nX_test dtypes:\n", X_test.dtypes)
print("Duplicates in X_train:", X_train.duplicated().sum())
print("Duplicates in X_test:", X_test.duplicated().sum())


overlapping_rows = X_test_sorted.merge(X_train_sorted, how="inner")
overlap_count = overlapping_rows.shape[0]
print("Overlap in Training & Testing Data:", overlap_count)



# Defining scoring metric for k-fold cross validation
def cv_scoring(estimator, X, y):
    return accuracy_score(y, estimator.predict(X))

# Initializing Models
models = {
    "SVC": SVC(),
    "Gaussian NB": GaussianNB(),
    "Random Forest": RandomForestClassifier(random_state=18)
}

# Producing cross validation score for the models
for model_name in models:
    model = models[model_name]
    scores = cross_val_score(model, X, y, cv=10, n_jobs=-1, scoring=cv_scoring)
    print("==" * 30)
    print(model_name)
    print(f"Scores: {scores}")
    print(f"Mean Score: {np.mean(scores)}")

# Training and testing SVM Classifier
svm_model = SVC()
svm_model.fit(X_train, y_train)
preds = svm_model.predict(X_test)

print(f"Accuracy on train data by SVM Classifier: {accuracy_score(y_train, svm_model.predict(X_train)) * 100}")
print(f"Accuracy on test data by SVM Classifier: {accuracy_score(y_test, preds) * 100}")

cf_matrix = confusion_matrix(y_test, preds)
plt.figure(figsize=(12, 8))
sns.heatmap(cf_matrix, annot=True)
plt.title("Confusion Matrix for SVM Classifier on Test Data")
plt.show()

# Training and testing Naive Bayes Classifier
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
preds = nb_model.predict(X_test)
print(f"Accuracy on train data by Naive Bayes Classifier: {accuracy_score(y_train, nb_model.predict(X_train)) * 100}")
print(f"Accuracy on test data by Naive Bayes Classifier: {accuracy_score(y_test, preds) * 100}")

cf_matrix = confusion_matrix(y_test, preds)
plt.figure(figsize=(12, 8))
sns.heatmap(cf_matrix, annot=True)
plt.title("Confusion Matrix for Naive Bayes Classifier on Test Data")
plt.show()

# Training and testing Random Forest Classifier
rf_model = RandomForestClassifier(random_state=18)
rf_model.fit(X_train, y_train)
preds = rf_model.predict(X_test)
print(f"Accuracy on train data by Random Forest Classifier: {accuracy_score(y_train, rf_model.predict(X_train)) * 100}")
print(f"Accuracy on test data by Random Forest Classifier: {accuracy_score(y_test, preds) * 100}")

cf_matrix = confusion_matrix(y_test, preds)
plt.figure(figsize=(12, 8))
sns.heatmap(cf_matrix, annot=True)
plt.title("Confusion Matrix for Random Forest Classifier on Test Data")
plt.show()

# Training the models on whole data
final_svm_model = SVC()
final_nb_model = GaussianNB()
final_rf_model = RandomForestClassifier(random_state=18)
final_svm_model.fit(X, y)
final_nb_model.fit(X, y)
final_rf_model.fit(X, y)

# Reading the test data
test_data = pd.read_csv("illnessnames/file1.csv").dropna(axis=1)

test_X = test_data.iloc[:, :-1]
test_Y = encoder.transform(test_data.iloc[:, -1])

# Making predictions by taking the mode of predictions
# made by all the classifiers
svm_preds = final_svm_model.predict(test_X)
nb_preds = final_nb_model.predict(test_X)
rf_preds = final_rf_model.predict(test_X)

# Use np.unique to get the mode of the predictions
def get_mode_prediction(predictions):
    unique_elements, counts_elements = np.unique(predictions, return_counts=True)
    return unique_elements[np.argmax(counts_elements)]

final_preds = [get_mode_prediction([i, j, k]) for i, j, k in zip(svm_preds, nb_preds, rf_preds)]

print(f"Accuracy on Test dataset by the combined model: {accuracy_score(test_Y, final_preds) * 100}")

cf_matrix = confusion_matrix(test_Y, final_preds)
plt.figure(figsize=(12, 8))
sns.heatmap(cf_matrix, annot=True)
plt.title("Confusion Matrix for Combined Model on Test Dataset")
plt.show()

# Symptom index dictionary and data_dict
symptoms = X.columns.values

# Creating a symptom index dictionary to encode the 
# input symptoms into numerical form 
symptom_index = {}
for index, value in enumerate(symptoms):
    symptom = " ".join([i.capitalize() for i in value.split("_")])
    symptom_index[symptom] = index

data_dict = {
    "symptom_index": symptom_index,
    "predictions_classes": encoder.classes_
}

# Defining the Function 
# Input: string containing symptoms separated by commas 
# Output: Generated predictions by models 
def predictDisease(symptoms): 
    symptoms = symptoms.lower().split(",")  # Convert to lowercase

    # creating input data for the models 
    input_data = [0] * len(data_dict["symptom_index"]) 
    for symptom in symptoms: 
        index = data_dict["symptom_index"].get(symptom.strip(), None)  # Remove spaces
        if index is not None:
            input_data[index] = 1

    # reshaping the input data and converting it into a suitable format for model predictions 
    input_data = pd.DataFrame([input_data], columns=X.columns)
    
    # Ensure models make predictions
    try:
        rf_prediction = data_dict["predictions_classes"][final_rf_model.predict(input_data)[0]]
        nb_prediction = data_dict["predictions_classes"][final_nb_model.predict(input_data)[0]]
        svm_prediction = data_dict["predictions_classes"][final_svm_model.predict(input_data)[0]]
        
        # making final prediction by taking mode of all predictions 
        final_prediction = get_mode_prediction([rf_prediction, nb_prediction, svm_prediction])
        predictions = { 
            "rf_model_prediction": rf_prediction, 
            "naive_bayes_prediction": nb_prediction, 
            "svm_model_prediction": svm_prediction, 
            "final_prediction": str(final_prediction) 
        } 
    except Exception as e:
        predictions = {"error": f"Prediction failed: {str(e)}"}

    return predictions 

# Function to take user input and predict disease
def get_user_input_and_predict():
    user_input = input("Enter symptoms separated by commas: ").strip()
    predictions = predictDisease(user_input)
    if "error" in predictions:
        print(predictions["error"])
    else:
        print("\n🔍 Predicted Disease Information:")
        print(f"✅ Random Forest Prediction: {predictions['rf_model_prediction']}")
        print(f"✅ Naive Bayes Prediction: {predictions['naive_bayes_prediction']}")
        print(f"✅ SVM Prediction: {predictions['svm_model_prediction']}")
        print(f"🚀 Final Prediction: {predictions['final_prediction']}")

# Get user input and predict disease
get_user_input_and_predict()








