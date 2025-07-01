import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder

# Load dataset
df = pd.read_csv('healthcare_data_updated.csv')

# Features
basic_features = ['age', 'bmi', 'bp_systolic', 'bp_diastolic', 'heart_rate', 'glucose', 'cholesterol', 'sleep_hours']
categorical_features = ['smoking', 'alcohol', 'exercise_level', 'gender', 'medical_history', 'stress_level']

all_features = basic_features + categorical_features
target = 'health_risk'




# Existing Inputs...
gender = st.sidebar.selectbox("Gender", ["Male", "Female", "Other"])
cholesterol = st.sidebar.number_input("Cholesterol Level (mg/dL)", 100, 300, 200)
medical_history = st.sidebar.selectbox("Family Medical History of Disease", ["Yes", "No"])
sleep_hours = st.sidebar.slider("Average Sleep Hours per Night", 0, 12, 7)
stress_level = st.sidebar.selectbox("Stress Level", ["Low", "Moderate", "High"])

# One-hot encode categorical columns
df_encoded = pd.get_dummies(df[all_features])
X = df_encoded
y = df[target]

# Train model
model = RandomForestClassifier()
model.fit(X, y)




# Recommendation function
# Update recommendation logic for stress or sleep
def generate_recommendation(risk_level, stress, sleep):
    advice = ""
    if risk_level == 'High':
        advice += "⚠️ High Risk: Immediate doctor consultation, strict diet, regular check-ups.\n"
    elif risk_level == 'Medium':
        advice += "⚠️ Moderate Risk: Improve exercise, control diet, monitor health.\n"
    else:
        advice += "✅ Low Risk: Maintain healthy lifestyle.\n"
    
    # Stress advice
    if stress == 'High':
        advice += "🧘 High Stress Detected: Practice meditation, reduce workload.\n"
    elif stress == 'Moderate':
        advice += "☁️ Moderate Stress: Consider light relaxation activities.\n"
    
    # Sleep advice
    if sleep < 6:
        advice += "💤 Poor Sleep: Aim for at least 7-8 hours of sleep.\n"
    
    return advice



# Diet Plan Function
def get_diet_plan(age):
    if age < 18:
        return "🍎 Diet Plan: Plenty of fruits, milk, lean proteins, avoid junk food."
    elif age < 40:
        return "🥗 Diet Plan: High fiber, complex carbs, fruits, vegetables, lean protein."
    elif age < 60:
        return "🍛 Diet Plan: Balanced diet with focus on heart health, nuts, moderate carbs."
    else:
        return "🍵 Diet Plan: Light meals, more fruits, calcium, vitamin D, and hydration."
# ------------------------
# Streamlit Interface
# ------------------------

st.title("🩺 Personalized Healthcare Recommendation")

st.sidebar.header("Enter Your Health Details:")

# Numeric Inputs
age = st.sidebar.number_input("Age", 1, 100, 30)
bmi = st.sidebar.number_input("BMI", 10.0, 50.0, 22.0)
bp_systolic = st.sidebar.number_input("Blood Pressure (Systolic)", 80, 200, 120)
bp_diastolic = st.sidebar.number_input("Blood Pressure (Diastolic)", 50, 150, 80)
heart_rate = st.sidebar.number_input("Heart Rate", 40, 200, 75)
glucose = st.sidebar.number_input("Glucose Level", 50, 300, 100)

# Categorical Inputs
smoking = st.sidebar.selectbox("Smoking Habit", ["Yes", "No"])
alcohol = st.sidebar.selectbox("Alcohol Consumption", ["None", "Moderate", "High"])
exercise_level = st.sidebar.selectbox("Exercise Level", ["Low", "Moderate", "High"])

# Prediction Button
if st.sidebar.button("Get Health Recommendation"):
    
    new_data = pd.DataFrame({
    'age': [age],
    'bmi': [bmi],
    'bp_systolic': [bp_systolic],
    'bp_diastolic': [bp_diastolic],
    'heart_rate': [heart_rate],
    'glucose': [glucose],
    'smoking': [smoking],
    'alcohol': [alcohol],
    'exercise_level': [exercise_level],
    'gender': [gender],
    'cholesterol': [cholesterol],
    'medical_history': [medical_history],
    'sleep_hours': [sleep_hours],
    'stress_level': [stress_level]
    }) 

    
    # One-hot encode new data to match training columns
    new_data_encoded = pd.get_dummies(new_data)
    
    # Align columns (add missing ones with 0)
    for col in X.columns:
        if col not in new_data_encoded.columns:
            new_data_encoded[col] = 0
    new_data_encoded = new_data_encoded[X.columns]
    
    # Predict
    risk = model.predict(new_data_encoded)[0]
    recommendation = generate_recommendation(risk,stress_level,sleep_hours)
    diet_plan = get_diet_plan(age)

    
    st.subheader(f"Predicted Health Risk: {risk}")
    st.success(recommendation)
    st.info(diet_plan)
