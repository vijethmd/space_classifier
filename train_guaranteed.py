import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
import joblib

print("🚀 Training Guaranteed Working Model...")

# Load data
df = pd.read_csv("star_classification.csv")

# Print all columns to see what we have
print("📋 All columns in your dataset:")
for col in df.columns:
    print(f"  - {col}")

# Use only the photometric filters (u, g, r, i, z) which should definitely exist
# These are standard SDSS filter names
features = [col for col in ['u', 'g', 'r', 'i', 'z'] if col in df.columns]

if not features:
    # If even those don't exist, use the first 5 numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    features = numeric_cols[:5]
    print(f"⚠️  Using numeric columns: {features}")

print(f"🎯 Using features: {features}")

X = df[features]
y = df['class']

print(f"🏷️ Classes: {y.unique()}")
print(f"📊 Class distribution:\n{y.value_counts()}")

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split and scale
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluate
accuracy = accuracy_score(y_test, model.predict(X_test_scaled))
print(f"🎯 Model Accuracy: {accuracy:.4f}")

# Save models
joblib.dump(model, 'random_forest_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(le, 'label_encoder.pkl')
joblib.dump(features, 'feature_names.pkl')

print("💾 All models saved successfully!")
print("🎉 Now run: streamlit run simple_app.py")