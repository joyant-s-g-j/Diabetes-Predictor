import pandas as pd
import pickle

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

# Load datasets
df = pd.read_csv("dataset/diabetes_prediction_dataset.csv")

# Outlier Detection and Removal
num_cols = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']
Q1 = df[num_cols].quantile(0.25)
Q3 = df[num_cols].quantile(0.75)

IQR = Q3 - Q1

df = df[~((df[num_cols] < (Q1 - 1.5 * IQR)) | (df[num_cols] > (Q3 + 1.5 * IQR))).any(axis=1)]

# Feature Engineering
df['health_risk'] = df['hypertension'] + df['heart_disease']

# Feature-Target Split
X = df.drop('diabetes', axis=1)
y = df['diabetes']

# Preprocessing Pipeline
num_features = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level', 'health_risk']
cat_features = ['gender', 'hypertension', 'heart_disease', 'smoking_history']

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_features),
    ('cat', OneHotEncoder(drop='first'), cat_features)
])

# Model Pipeline
model = RandomForestClassifier(
  n_estimators=50,
  max_depth=10,
  min_samples_split=2,
  random_state=42,
  n_jobs=-1,
)

# Full Pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', model)
])

#  Train-Test-Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train the model
pipeline.fit(X_train, y_train)

# Evaluate the model
y_pred = pipeline.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Save the trained model
with open("diabetes_prediction_model.pkl", "wb") as file:
  pickle.dump(pipeline, file)

print("Model saved as diabetes_prediction_model.pkl")