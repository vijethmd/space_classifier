import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib

print("🚀 Training Improved Random Forest Model...")

# Load data
df = pd.read_csv("star_classification.csv")

print("📊 Dataset Info:")
print(f"Shape: {df.shape}")
print(f"Classes: {df['class'].value_counts()}")

# Use the most important features for astronomy
# Photometric colors are often more important than individual magnitudes
features = ['u', 'g', 'r', 'i', 'z', 'redshift']

X = df[features]
y = df['class']

# Add some engineered features (common in astronomy)
X_engineered = X.copy()
X_engineered['u_g'] = X['u'] - X['g']  # Color index
X_engineered['g_r'] = X['g'] - X['r']  # Color index  
X_engineered['r_i'] = X['r'] - X['i']  # Color index
X_engineered['i_z'] = X['i'] - X['z']  # Color index

print(f"🎯 Using features: {X_engineered.columns.tolist()}")

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_engineered, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train improved Random Forest
print("\n🌲 Training Improved Random Forest...")
improved_model = RandomForestClassifier(
    n_estimators=200,           # More trees
    max_depth=20,               # Deeper trees
    min_samples_split=5,        # Better generalization
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

improved_model.fit(X_train_scaled, y_train)

# Evaluate
y_pred = improved_model.predict(X_test_scaled)
y_pred_proba = improved_model.predict_proba(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
print(f"🎯 Improved Model Accuracy: {accuracy:.4f}")

print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Check confidence scores
confidence_scores = np.max(y_pred_proba, axis=1)
print(f"\n📈 Confidence Statistics:")
print(f"Average confidence: {np.mean(confidence_scores):.4f}")
print(f"Min confidence: {np.min(confidence_scores):.4f}")
print(f"Max confidence: {np.max(confidence_scores):.4f}")

# Save improved models
joblib.dump(improved_model, 'random_forest_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(le, 'label_encoder.pkl')
joblib.dump(X_engineered.columns.tolist(), 'feature_names.pkl')

print("💾 Improved models saved successfully!")

# Test with some sample data
print("\n🧪 Sample Predictions:")
sample_data = [
    [19.0, 18.5, 17.2, 16.8, 16.5, 0.1],  # Likely star
    [18.5, 17.8, 17.2, 16.9, 16.7, 0.05], # Likely galaxy  
    [17.8, 17.5, 17.3, 17.2, 17.1, 1.5]   # Likely quasar
]

for i, sample in enumerate(sample_data):
    # Add engineered features
    sample_df = pd.DataFrame([sample], columns=features)
    sample_df['u_g'] = sample_df['u'] - sample_df['g']
    sample_df['g_r'] = sample_df['g'] - sample_df['r']
    sample_df['r_i'] = sample_df['r'] - sample_df['i']
    sample_df['i_z'] = sample_df['i'] - sample_df['z']
    
    sample_scaled = scaler.transform(sample_df)
    pred = improved_model.predict(sample_scaled)[0]
    proba = improved_model.predict_proba(sample_scaled)[0]
    confidence = np.max(proba)
    
    print(f"Sample {i+1}: {le.inverse_transform([pred])[0]} (confidence: {confidence:.4f})")

print("\n🎉 Improved training complete! Run: streamlit run improved_app.py")