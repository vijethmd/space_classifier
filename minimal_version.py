import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib

print("🚀 Starting Minimal Training Version...")

try:
    # Load data
    df = pd.read_csv("star_classification.csv")
    print(f"✅ Data loaded: {df.shape}")
    
    # Prepare features
    features = ['u', 'g', 'r', 'i', 'z', 'redshift', 'plate', 'MJD', 'fiberID']
    X = df[features]
    y = df['class']
    
    print(f"Classes: {y.unique()}")
    print(f"Class distribution:\n{y.value_counts()}")

    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train Random Forest
    print("Training Random Forest...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_train_scaled, y_train)

    # Evaluate
    y_pred = rf_model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"🎯 Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # Save models
    joblib.dump(rf_model, 'random_forest_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(le, 'label_encoder.pkl')
    joblib.dump(features, 'feature_names.pkl')

    print("💾 Models saved successfully!")
    print("\n🎉 Now run: streamlit run simple_app.py")

except Exception as e:
    print(f"❌ Error: {e}")
    print("Make sure the 'star_classification.csv' file is in the same directory")