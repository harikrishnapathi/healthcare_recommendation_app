import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Load dataset
df = pd.read_csv('healthcare_data_1000.csv')

print("Initial Data Shape:", df.shape)

# ------------------------
# 1. Handle Missing Values
# ------------------------

# Check for missing values
print("\nMissing values per column:")
print(df.isnull().sum())

# For simplicity, fill missing numerical values with the mean, categorical with the most frequent
numerical_cols = ['age', 'bmi', 'heart_rate', 'glucose']
categorical_cols = ['gender', 'smoking', 'alcohol', 'exercise_level', 'cholesterol', 'medical_history', 'medication']

# ------------------------
# 2. Separate Features and Target
# ------------------------

X = df.drop(columns=['health_risk', 'recommendation'])
y = df['health_risk']

# ------------------------
# 3. Preprocessing Pipelines
# ------------------------

# Pipeline for numerical features: Impute missing values & scale
numerical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# Pipeline for categorical features: Impute missing values & encode
categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

# Combine pipelines
preprocessor = ColumnTransformer([
    ('num', numerical_pipeline, numerical_cols),
    ('cat', categorical_pipeline, categorical_cols)
])

# ------------------------
# 4. Train-Test Split
# ------------------------

# 70% Train, 15% Validation, 15% Test
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.176, random_state=42)  # 15% of total

print("\nData Shapes:")
print(f"Train: {X_train.shape}, Validation: {X_val.shape}, Test: {X_test.shape}")

# ------------------------
# 5. Apply Preprocessing
# ------------------------

# Fit and transform training data, transform validation and test sets
X_train_processed = preprocessor.fit_transform(X_train)
X_val_processed = preprocessor.transform(X_val)
X_test_processed = preprocessor.transform(X_test)

print("\nPreprocessing completed.")
print("Processed feature shape:", X_train_processed.shape)
