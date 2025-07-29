import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pickle

# Load dataset
data = pd.read_csv('Salary Data.csv')

# Check and drop rows with missing target or critical features
data = data.dropna(subset=['Salary', 'Gender', 'Job Title', 'Age', 'Years of Experience'])

# Encode categorical variables: Job Title, Gender, Education Level if present
categorical_cols = []
for col in ['Job Title', 'Gender', 'Education Level']:
    if col in data.columns:
        categorical_cols.append(col)

data_encoded = pd.get_dummies(data, columns=categorical_cols, drop_first=True)

# Ensure all features are numeric
X = data_encoded.drop('Salary', axis=1).select_dtypes(include=['number'])
y = data_encoded['Salary']

# Final check for NaNs
if X.isnull().sum().sum() > 0:
    print("⚠️ Warning: NaNs detected in features. Filling with 0.")
    X = X.fillna(0)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Save model
with open('salary_model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("✅ Model trained and saved successfully.")
