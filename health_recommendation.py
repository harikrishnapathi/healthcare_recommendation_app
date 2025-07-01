import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Load your updated dataset
df = pd.read_csv('healthcare_data_updated.csv')

# Select features
features = ['age', 'bmi', 'bp_systolic', 'bp_diastolic', 'heart_rate', 'glucose']
X = df[features]
y = df['health_risk']

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# ------------------------
# Recommendation Function
# ------------------------

def generate_recommendation(risk_level):
    if risk_level == 'High':
        return "⚠️ High Risk: Immediate doctor consultation, strict diet, regular check-ups recommended."
    elif risk_level == 'Medium':
        return "⚠️ Moderate Risk: Improve exercise routine, control diet, monitor health."
    else:
        return "✅ Low Risk: Maintain healthy lifestyle with regular exercise and balanced diet."

# ------------------------
# Predict for New Patient
# ------------------------

# Example new patient data (change values to test)
new_patient = pd.DataFrame({
    'age': [55],
    'bmi': [29],
    'bp_systolic': [145],
    'bp_diastolic': [95],
    'heart_rate': [85],
    'glucose': [130]
})
new_patient = pd.DataFrame({
    'age': [35],
    'bmi': [22],
    'bp_systolic': [120],
    'bp_diastolic': [80],
    'heart_rate': [75],
    'glucose': [90]
})


# Predict risk
predicted_risk = model.predict(new_patient)[0]
print(f"Predicted Health Risk: {predicted_risk}")

# Get recommendation
recommendation = generate_recommendation(predicted_risk)
print("Personalized Recommendation:")
print(recommendation)
