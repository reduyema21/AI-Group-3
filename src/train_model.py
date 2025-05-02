# train_model.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load the dataset
data_path = 'Data/water_quality.csv'  # Fixed backslash issue for cross-platform compatibility
if not os.path.exists(data_path):
    raise FileNotFoundError(f"Dataset not found at {data_path}")

df = pd.read_csv(data_path)

# 2. Basic preprocessing
df.dropna(inplace=True)

# Split features and label (assumes label is last column)
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Show label distribution
print("Label distribution:")
print(y.value_counts())

# If labels are non-numeric, encode them
if y.dtype == 'object':
    y_encoded, uniques = pd.factorize(y)
    print(f"Encoded labels: {dict(enumerate(uniques))}")
    y = y_encoded

# 3. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 4. Train Random Forest with class balancing
model = RandomForestClassifier(
    n_estimators=100,
    class_weight='balanced',
    random_state=42
)
model.fit(X_train, y_train)

# 5. Evaluate
y_pred = model.predict(X_test)
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 6. Confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# 7. Save the model
joblib.dump(model, 'water_quality_model.pkl')
print("Model saved to water_quality_model.pkl")
