import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

# ------------------------
# 1. Load Dataset & Preprocessing
# ------------------------

df = pd.read_csv('healthcare_data_1000.csv')

numerical_cols = ['age', 'bmi', 'heart_rate', 'glucose']
categorical_cols = ['gender', 'smoking', 'alcohol', 'exercise_level', 'cholesterol', 'medical_history', 'medication']

X = df.drop(columns=['health_risk', 'recommendation'])
y = df['health_risk']

# Preprocessing pipelines
numerical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', numerical_pipeline, numerical_cols),
    ('cat', categorical_pipeline, categorical_cols)
])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_processed, y_train)

# ------------------------
# 2. Recommendation Function
# ------------------------

def generate_recommendation(risk_level):
    if risk_level == 'High':
        return "High risk detected: Consult a doctor, adopt strict diet, regular health check-ups."
    elif risk_level == 'Medium':
        return "Moderate risk: Improve exercise routine, monitor diet, reduce alcohol/smoking if applicable."
    else:
        return "Low risk: Maintain balanced lifestyle with regular exercise and healthy diet."

# ------------------------
# 3. Test with New Patient Data
# ------------------------

# Example new patient details (modify as needed)
new_patient = pd.DataFrame({
    'age': [50],
    'gender': ['Male'],
    'smoking': ['Yes'],
    'alcohol': ['Moderate'],
    'exercise_level': ['Low'],
    'bmi': [29],
    'blood_pressure': ['150/95'],  # Can drop if not used in model
    'heart_rate': [85],
    'cholesterol': ['High'],
    'glucose': [135],
    'medical_history': ['Hypertension'],
    'medication': ['Beta-blockers']
})

# Drop columns not used in the model
new_patient = new_patient[X.columns]

# Preprocess new patient data
new_patient_processed = preprocessor.transform(new_patient)

# Predict risk level
predicted_risk = model.predict(new_patient_processed)[0]
print(f"\nPredicted Health Risk: {predicted_risk}")

# Get recommendation
recommendation = generate_recommendation(predicted_risk)
print(f"Personalized Recommendation: {recommendation}")
