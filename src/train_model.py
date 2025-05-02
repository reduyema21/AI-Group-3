# train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# 1. Load the dataset
data_path = 'Data\water_quality.csv'
if not os.path.exists(data_path):
    raise FileNotFoundError(f"Dataset not found at {data_path}")

df = pd.read_csv(data_path)

# 2. Basic preprocessing
# Drop rows with missing values (or handle differently if needed)
df.dropna(inplace=True)

# Assuming the last column is the label (adjust if needed)
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# If labels are not numeric, convert them (optional)
if y.dtype == 'object':
    y = pd.factorize(y)[0]

# 3. Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Train a Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. Evaluate the model
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# 6. Save the trained model
joblib.dump(model, 'water_quality_model.pkl')
print("Model saved to water_quality_model.pkl")
