import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('healthcare_data_1000.csv')

# Preview first few rows
print("\nFirst 5 records:")
print(df.head())

# General info
print("\nDataset Info:")
print(df.info())

# Descriptive statistics
print("\nDescriptive Statistics:")
print(df.describe(include='all'))

# Count of unique values for categorical columns
categorical_cols = ['gender', 'smoking', 'alcohol', 'exercise_level', 'cholesterol', 'medical_history', 'medication', 'health_risk']
for col in categorical_cols:
    print(f"\nValue counts for {col}:")
    print(df[col].value_counts())

# Distribution of Age
plt.figure(figsize=(8, 4))
sns.histplot(df['age'], bins=20, kde=True)
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()

# Distribution of BMI by Health Risk
plt.figure(figsize=(8, 4))
sns.boxplot(x='health_risk', y='bmi', data=df)
plt.title('BMI Distribution by Health Risk')
plt.show()

# Count Plot of Health Risk Categories
plt.figure(figsize=(6, 4))
sns.countplot(x='health_risk', data=df)
plt.title('Health Risk Distribution')
plt.show()

# Pairplot for Numeric Relationships
sns.pairplot(df, hue='health_risk', vars=['age', 'bmi', 'heart_rate', 'glucose'])
plt.show()

# Correlation Heatmap for Numeric Features
numeric_df = df[['age', 'bmi', 'heart_rate', 'glucose']].copy()
corr = numeric_df.corr()

plt.figure(figsize=(6, 4))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()
