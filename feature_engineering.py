import pandas as pd

# Load your dataset
df = pd.read_csv('healthcare_data_1000.csv')

# Split blood pressure into systolic and diastolic
df['bp_systolic'] = df['blood_pressure'].str.split('/').str[0].astype(int)
df['bp_diastolic'] = df['blood_pressure'].str.split('/').str[1].astype(int)

# Create BMI category
df['bmi_category'] = pd.cut(df['bmi'], bins=[0, 18.5, 24.9, 29.9, 100], labels=['Underweight', 'Normal', 'Overweight', 'Obese'])

# Show the new features
print(df[['blood_pressure', 'bp_systolic', 'bp_diastolic', 'bmi', 'bmi_category']].head())

# Optional: Save updated dataset
df.to_csv('healthcare_data_updated.csv', index=False)
print("\nUpdated dataset saved as 'healthcare_data_updated.csv'")
