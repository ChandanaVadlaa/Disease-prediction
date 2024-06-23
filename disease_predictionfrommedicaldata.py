import numpy as np
import csv
from random import randint, choice
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

# Create a synthetic dataset and save to CSV
def create_synthetic_dataset(filename):
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = ['age', 'gender', 'symptom1', 'symptom2', 'symptom3', 'history_of_disease', 'disease']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for _ in range(1000):
            writer.writerow({
                'age': randint(20, 80),
                'gender': choice(['male', 'female']),
                'symptom1': randint(0, 1),
                'symptom2': randint(0, 1),
                'symptom3': randint(0, 1),
                'history_of_disease': randint(0, 1),
                'disease': choice(['disease_A', 'disease_B', 'no_disease'])
            })

# Load dataset from CSV
def load_dataset(filename):
    data = []
    with open(filename, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            data.append(row)
    return data

# Preprocess the data
def preprocess_data(data):
    features = []
    labels = []
    for row in data:
        features.append([
            int(row['age']),
            row['gender'],
            int(row['symptom1']),
            int(row['symptom2']),
            int(row['symptom3']),
            int(row['history_of_disease'])
        ])
        labels.append(row['disease'])
    
    # Encode categorical data
    le_gender = LabelEncoder()
    le_gender.fit(['male', 'female'])
    le_disease = LabelEncoder()
    le_disease.fit(['disease_A', 'disease_B', 'no_disease'])
    
    for feature in features:
        feature[1] = le_gender.transform([feature[1]])[0]
    
    labels = le_disease.transform(labels)
    
    return np.array(features), np.array(labels), le_gender, le_disease

# Create synthetic dataset
create_synthetic_dataset('synthetic_medical_data.csv')

# Load and preprocess data
data = load_dataset('synthetic_medical_data.csv')
X, y, le_gender, le_disease = preprocess_data(data)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Decision Tree Classifier
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Save the model and encoders using pickle
import pickle
with open('disease_prediction_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)
with open('label_encoder_gender.pkl', 'wb') as le_gender_file:
    pickle.dump(le_gender, le_gender_file)
with open('label_encoder_disease.pkl', 'wb') as le_disease_file:
    pickle.dump(le_disease, le_disease_file)

# Function to predict disease for a new patient
def predict_disease(patient_data):
    # Load the model and encoders
    with open('disease_prediction_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('label_encoder_gender.pkl', 'rb') as le_gender_file:
        le_gender = pickle.load(le_gender_file)
    with open('label_encoder_disease.pkl', 'rb') as le_disease_file:
        le_disease = pickle.load(le_disease_file)
    
    # Encode the gender
    patient_data['gender'] = le_gender.transform([patient_data['gender']])[0]
    
    # Convert to DataFrame
    patient_features = np.array([[patient_data['age'], patient_data['gender'], patient_data['symptom1'], patient_data['symptom2'], patient_data['symptom3'], patient_data['history_of_disease']]])
    
    # Predict
    prediction = model.predict(patient_features)
    disease = le_disease.inverse_transform(prediction)
    
    return disease[0]

# Example usage
new_patient = {
    'age': 50,
    'gender': 'female',
    'symptom1': 1,
    'symptom2': 0,
    'symptom3': 1,
    'history_of_disease': 0
}

predicted_disease = predict_disease(new_patient)
print(f"Predicted disease: {predicted_disease}")
