import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load your dataset
df = pd.read_csv('healthcare_data_updated.csv')

# Select features for training (keep important ones)
features = ['age', 'bmi', 'bp_systolic', 'bp_diastolic', 'heart_rate', 'glucose']
X = df[features]

# Target (what we want to predict)
y = df['health_risk']

# Split into train (80%) and test (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create Random Forest model
model = RandomForestClassifier()

# Train the model
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Calculate accuracy
acc = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {acc:.2f}")

# Show few predictions
results = X_test.copy()
results['Actual Risk'] = y_test
results['Predicted Risk'] = y_pred
print(results.head())

