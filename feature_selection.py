import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load updated dataset
df = pd.read_csv('healthcare_data_updated.csv')

# Show columns
print("\nAll Features:")
print(df.columns.tolist())

# Let's see correlation with health risk (encode risk for numeric calculation)
df['risk_encoded'] = df['health_risk'].map({'Low': 0, 'Medium': 1, 'High': 2})

# Check correlation of numeric features with risk
correlation = df.corr(numeric_only=True)
print("\nFeature Correlation with Health Risk:")
print(correlation['risk_encoded'].sort_values(ascending=False))

# Optional: Visualize correlations
sns.heatmap(correlation, annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()
